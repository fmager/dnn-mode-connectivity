
import argparse
import numpy as np

import jax.numpy as jnp
import jax
from jax import random
from jax import jit
from jax.scipy.stats import norm

from gp import GaussianProcessRegression
from gp import SquaredExponentialKernel
from gp import Matern32Kernel
from gp import Hyperparameters


import os
import tabulate
import torch
import torch.nn.functional as F
import tqdm

import data
import models
import curves
import utils
import measures

jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser(description='Relative representation evaluation')
parser.add_argument('--dir', type=str, default='/tmp/eval', metavar='DIR',
                help='result directory (default: /tmp/eval)')
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: VGG16)')
parser.add_argument('--seed_from_to', type=str, default='0-1', metavar='SEED',
                    help='seed range (default: 0-1)')
parser.add_argument('--curve', type=str, default='PolyChain', metavar='CURVE',
                    help='curve type to use (default: PolyChain)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')
parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to eval (default: None)')
parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default='./data', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=2, metavar='N',
                    help='number of workers (default: 2)')
parser.add_argument('--num_points', type=int, default=61, metavar='N',
                    help='number of points on the curve (default: 61)')
parser.add_argument('--num_anchors', type=int, default=512, metavar='N',
                    help='number of anchors (default: 512)')
parser.add_argument('--projection', type=str, default='cosine', metavar='TYPE',
                    help='relative projection type (default: cosine)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--sampling_method', type=str, default='linear', metavar='METHOD',
                    help='sampling method for t (default: linear)')
parser.add_argument('--layer_name', type=str, default='', metavar='LAYER',
                    help='layer to evaluate')
parser.add_argument('--center', action='store_true', default=False,
                    help='center the latent space before projection')
parser.add_argument('--standardize', action='store_true', default=False,
                    help='standardize latent space before projection')


args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args.dir = os.path.join(args.dir, args.dataset.lower(), args.model.lower(), f'seed_{args.seed_from_to.replace('-','_to_')}', args.curve.lower())
args.ckpt = os.path.join(args.dir, args.ckpt)
os.makedirs(args.dir, exist_ok=True)

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test,
    shuffle_train=False,
    drop_last=False
)

targets = []
for _, t in loaders['test']:
    targets.append(t)
targets = torch.cat(targets, dim=0)

indices = torch.randperm(len(loaders['test'].dataset))[:args.num_anchors]
subset = torch.utils.data.Subset(loaders['test'].dataset, indices)

# Create the DataLoader
loaders['anchors'] = torch.utils.data.DataLoader(
    subset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers
)

architecture = getattr(models, args.model)
curve = getattr(curves, args.curve)
model = curves.CurveNet(
    num_classes,
    curve,
    architecture.curve,
    args.num_bends,
    architecture_kwargs=architecture.kwargs,
)
model.to(device)
checkpoint = torch.load(args.ckpt)
model.load_state_dict(checkpoint['model_state'])

criterion = F.cross_entropy
regularizer = curves.l2_regularizer(args.wd)

if args.projection == 'cosine':
    proj_fn = utils.cosine_projection
elif args.projection == 'euclidean':
    proj_fn = utils.euclidean_projection
elif args.projection == 'basis_norm':
    proj_fn = utils.basis_norm_projection
else:
    raise ValueError('Unknown projection type: %s' % args.projection)

rel_proj = utils.RelProjector(proj_fn, center=args.center, standardize=args.standardize)

@jit
def ei_acquisition(mu, var, best_value):
    """ implements the expected improvement acquisition function.
        The argument mu and var are the posterior mean and variance, respectively, 
        best_value denotes the best value M_N observed so far
    """
    diff = mu.ravel() - best_value
    std_f = jnp.sqrt(var)
    gamma = diff/std_f
    return std_f*(gamma*norm.cdf(gamma) + norm.pdf(gamma))

class ObjectiveFunction(object):
    """ Objective function for Bayesian optimization """
    def __init__(self, model, loaders, criterion, args, rel_proj, regularizer, targets, num_classes):
        self.model = model
        self.loaders = loaders
        self.criterion = criterion
        self.args = args
        self.rel_proj = rel_proj
        self.regularizer = regularizer
        self.previous_weights = None
        self.targets = targets
        self.predictions_sum = np.zeros((targets.size(0), num_classes))
        self.previous_weights = None

        self.ens_acc = []
        self.ts = []
        self.t = torch.FloatTensor([0.0]).cuda()
        self.test_acc = []
        self.test_err = []
        self.test_loss = []
        self.test_nll = []
        self.dl = []
        self.iteration = 0

    def get_status(self):
        i = self.iteration
        columns = ['i', 't', 'test acc', 'ens_acc']
        values = [self.iteration, self.ts[i], self.test_acc[i], self.ens_acc[i]]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')

        if i % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

        self.iteration += 1

    def query_y_given_t(self, t_value):


        self.ts.append(t_value.item())

        self.t.data.fill_(t_value.item())

        weights = self.model.weights(self.t)
        if self.previous_weights is not None:
            self.dl.append(np.sqrt(np.sum(np.square(weights - self.previous_weights))))
        self.previous_weights = weights.copy()

        utils.update_bn(self.loaders['train'], self.model, t=self.t)

        test_res = utils.test_relative(self.loaders, self.model, self.criterion, self.layer_name, self.rel_proj, self.regularizer, t=self.t)
        
        self.test_loss.append(test_res['loss'])
        self.test_nll.append(test_res['nll'])
        self.test_acc.append(test_res['accuracy'])
        self.test_err.append(100.0 - test_res['accuracy'])
        
        predictions = test_res['predictions']
        self.predictions_sum += predictions
        self.ens_acc.append(100.0 * np.mean(np.argmax(self.predictions_sum, axis=1) == targets.numpy()))

        z_input = test_res['output_abs']
        z_proj = test_res['output_rel']

        if self.abs_latent_samples == None:
            self.rel_latent_samples = z_proj.unsqueeze(-1).cpu()
            self.abs_latent_samples = z_input.unsqueeze(-1).cpu()
        else:
            self.rel_latent_samples = torch.cat((self.rel_latent_samples, z_proj.unsqueeze(-1).cpu()), dim=2)
            self.abs_latent_samples = torch.cat((self.abs_latent_samples, z_input.unsqueeze(-1).cpu()), dim=2)

        # Get the gradient of the FLD criterion with respect to ts
        crit, _, _ = measures.sample_fld(self.rel_latent_samples)

        

        self.get_status()

        return self.ens_acc[-1]


class BayesianOptimization(object):

    def __init__(self, t0, y0, data_generator, t_grid, hyperparameters, acq_fun, kernel=SquaredExponentialKernel()):
        self.data_generator = data_generator
        self.query_fun = data_generator.query_y_given_t
        self.t0, self.y0 = t0, y0
        self.x, self.y = np.copy(t0), np.copy(y0)
        self.t_grid = t_grid
        self.acq_fun = acq_fun

        # surrogate model
        self.hyperparameters = hyperparameters
        self.kernel = kernel
        self.update_surrogate()

    def update_surrogate(self):
        """ fit surrogate model using current data set (self.x, self.y) """
        self.model = GaussianProcessRegression(self.x, self.y, self.kernel, self.hyperparameters)
        self.mu_f, self.var_f = self.model.predict_f(self.x)

    
    def evaluate_acquisition_function(self):
        """ evaluate the acqusiition function """
        mu_fstar, var_fstar = self.model.predict_f(self.t_grid)
        return self.acq_fun(mu_fstar, var_fstar, jnp.max(self.mu_f))

    def next_iteration(self):

        # identify next query by evaluating acquisition function
        self.acq_val = self.evaluate_acquisition_function()
        x_new = jnp.atleast_2d(self.t_grid[jnp.argmax(self.acq_val)])

        # query new point and append
        y_new = self.query_fun(x_new)
        self.x, self.y = jnp.vstack((self.x, x_new)), jnp.vstack((self.y, y_new))

        # update model
        self.update_surrogate()



T = args.num_points

key = random.key(1235)
sigma_y = 0.15

obj_fun = ObjectiveFunction(model, loaders, criterion, args, rel_proj, regularizer, targets, num_classes)

t_grid = np.linspace(0, 1, 100)[:, None]

# sample t0 randomly
t0 = np.random.rand(2)[:, None]
y0 = np.array([0, 0])[:, None]

for i, t in enumerate(t0):
    y0[i] = obj_fun.query_y_given_t(t)

hyperparameters = Hyperparameters(kappa=2.0, lengthscale=1., sigma=sigma_y)
bo = BayesianOptimization(t0, y0, obj_fun, t_grid, hyperparameters, acq_fun=ei_acquisition)

while obj_fun.iteration < T:
    bo.next_iteration()
