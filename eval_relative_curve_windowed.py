
import argparse
import numpy as np
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

parser = argparse.ArgumentParser(description='Relative representation evaluation')
parser.add_argument('--dir', type=str, default='/tmp/eval', metavar='DIR',
                help='result directory (default: /tmp/eval)')
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: VGG16)')
parser.add_argument('--curve', type=str, default='PolyChain', metavar='CURVE',
                    help='curve type to use (default: PolyChain)')
parser.add_argument('--seed_from_to', type=str, default='0-1', metavar='SEED',
                    help='seed range (default: 0-1)')
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
parser.add_argument('--dt', type=float, default=0.01, metavar='DT',
                    help='delta time step t  (default: 0.01)')
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

def sample_t(num_points, method: str = 'linear'):
    '''
    Sample t values from the interval [0, 1]
    '''
    if method == 'linear':
        return np.linspace(args.dt/2, 1.0-(args.dt/2), num_points)
    elif method == 'normal':
        return np.sort(np.random.normal(0.5, 0.25, num_points))
    elif method == 'uniform':
        return np.sort(np.random.uniform(args.dt/2, 1.0-(args.dt/2), num_points))
    else:
        raise ValueError('Unknown method: %s' % method)

T = args.num_points
ts = sample_t(T, args.sampling_method)

rel_fld_sample = np.zeros(T)
rel_fld_class = np.zeros(T) 
abs_fld_sample = np.zeros(T)
abs_fld_class = np.zeros(T) 

test_loss = np.zeros(T)
test_nll = np.zeros(T)
test_acc = np.zeros(T)
test_err = np.zeros(T)

ens_acc = np.zeros(T)
ens_size = 0

dl = np.zeros(T)

columns = ['i', 't', 'rel fld sample', 'rel fld class', 'abs fld sample', 'abs fld class', 'test acc', 'ens_acc']
previous_weights = None
t = torch.FloatTensor([0.0]).cuda()

for i, t_value in enumerate(ts):

    predictions_sum = np.zeros((targets.size(0), num_classes))


    for j, t_dt in enumerate(np.arange(t_value-(args.dt/2), t_value, t_value-(args.dt/2))):

        t.data.fill_(t_value)
        weights = model.weights(t)
        if previous_weights is not None:
            dl[i] = np.sqrt(np.sum(np.square(weights - previous_weights)))
        previous_weights = weights.copy()

        utils.update_bn(loaders['train'], model, t=t)

        test_res = utils.test_relative(loaders, model, criterion, args.layer_name, rel_proj, regularizer, t=t)
    
        test_loss[i] += test_res['loss']
        test_nll[i] += test_res['nll']
        test_acc[i] += test_res['accuracy']
        test_err[i] += 100.0 - test_acc[i]

        predictions = test_res['predictions']
        predictions_sum += predictions

    ens_acc[i] += 100.0 * np.mean(np.argmax(predictions_sum, axis=1) == targets.numpy())

    z_input = test_res['output_abs']
    z_proj = test_res['output_rel']

    if i == 0:
        rel_latent_samples = z_proj.unsqueeze(-1).cpu()
        abs_latent_samples = z_input.unsqueeze(-1).cpu()
    else:
        rel_latent_samples = torch.cat((rel_latent_samples, z_proj.unsqueeze(-1).cpu()), dim=2)
        abs_latent_samples = torch.cat((abs_latent_samples, z_input.unsqueeze(-1).cpu()), dim=2)

    crit, _, _ = measures.sample_fld(rel_latent_samples)
    rel_fld_sample[i] = crit
    crit, _, _ = measures.class_fld(rel_latent_samples, targets)
    rel_fld_class[i] = crit

    crit, _, _ = measures.sample_fld(abs_latent_samples)
    abs_fld_sample[i] = crit
    crit, _, _ = measures.class_fld(abs_latent_samples, targets)
    abs_fld_class[i] = crit

    values = [i, t, rel_fld_sample[i], rel_fld_class[i], abs_fld_sample[i], abs_fld_class[i], test_acc[i], ens_acc[i]]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')

    if i % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)


def stats(values, dl):
    # remove None values
    values = values[values != None]
    min = np.min(values)
    max = np.max(values)
    avg = np.mean(values)
    int = np.sum(0.5 * (values[:-1] + values[1:]) * dl[1:]) / np.sum(dl[1:])
    return min, max, avg, int


test_loss_min, test_loss_max, test_loss_avg, test_loss_int = stats(test_loss, dl)
test_nll_min, test_nll_max, test_nll_avg, test_nll_int = stats(test_nll, dl)
test_err_min, test_err_max, test_err_avg, test_err_int = stats(test_err, dl)

print('\nLength: %.2f\n' % np.sum(dl))
print(tabulate.tabulate([
        ['test nll', test_nll[0], test_nll[-1], test_nll_min, test_nll_max, test_nll_avg, test_nll_int],
        ['test error (%)', test_err[0], test_err[-1], test_err_min, test_err_max, test_err_avg, test_err_int],
        ['rel fld sample', rel_fld_sample[0], rel_fld_sample[-1], np.min(rel_fld_sample), np.max(rel_fld_sample), np.mean(rel_fld_sample), np.sum(rel_fld_sample)],
        ['rel fld class', rel_fld_class[0], rel_fld_class[-1], np.min(rel_fld_class), np.max(rel_fld_class), np.mean(rel_fld_class), np.sum(rel_fld_class)],
        ['abs fld sample', abs_fld_sample[0], abs_fld_sample[-1], np.min(abs_fld_sample), np.max(abs_fld_sample), np.mean(abs_fld_sample), np.sum(abs_fld_sample)],
        ['abs fld class', abs_fld_class[0], abs_fld_class[-1], np.min(abs_fld_class), np.max(abs_fld_class), np.mean(abs_fld_class), np.sum(abs_fld_class)],
    ], [
        '', 'start', 'end', 'min', 'max', 'avg', 'int'
    ], tablefmt='simple', floatfmt='10.4f'))
    
np.savez(
    os.path.join(args.dir, f'{args.sampling_method}_{args.projection}_rel_eval.npz'),
    ts=ts,
    rel_fld_class=rel_fld_class,
    rel_fld_sample=rel_fld_sample,
    abs_fld_class=abs_fld_class,
    abs_fld_sample=abs_fld_sample,
    test_acc=test_acc,
    ens_acc=ens_acc,
)

