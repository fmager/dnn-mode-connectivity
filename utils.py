import numpy as np
import os
import torch
import torch.nn.functional as F

import curves


def l2_regularizer(weight_decay):
    def regularizer(model):
        l2 = 0.0
        for p in model.parameters():
            l2 += torch.sqrt(torch.sum(p ** 2))
        return 0.5 * weight_decay * l2
    return regularizer


def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def train(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input)
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }


def test(test_loader, model, criterion, regularizer=None, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0
    preds = []

    model.eval()

    for input, target in test_loader:
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input, **kwargs)
        probs = F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        nll = criterion(output, target)
        loss = nll.clone()
        if regularizer is not None:
            loss += regularizer(model)

        nll_sum += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
        'predictions': np.vstack(preds),
    }

def test_relative(test_loader, model, criterion, layer_name: str, rel_proj, regularizer=None, **kwargs):

    outputs_abs = []
    outputs_rel = []

    latent = torch.Tensor()
    def forward_hook(module, input, output):
        nonlocal latent
        latent = input[0].clone().detach()
    
    layer = getattr(model.net, layer_name)
    handle = layer.register_forward_hook(forward_hook)

    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0
    preds = []

    model.eval()

    for input, target in test_loader['test']:

        input = input.cuda(non_blocking=True).requires_grad_(False)
        target = target.cuda(non_blocking=True).requires_grad_(False)

        out = model(input, **kwargs)

        outputs_abs.append(latent.clone().detach().cpu())

        probs = F.softmax(out, dim=1)
        preds.append(probs.cpu().data.numpy())
        nll = criterion(out, target)
        loss = nll.clone()
        if regularizer is not None:
            loss += regularizer(model)

        nll_sum += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
        pred = out.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    abs_latent = torch.cat(outputs_abs, dim=0)
    
    for input, target in test_loader['anchors']:
        input = input.cuda(non_blocking=True).requires_grad_(False)
        target = target.cuda(non_blocking=True).requires_grad_(False)

        _ = model(input, **kwargs)
        anchors = latent.clone().detach().cpu()
    
        rel_latent = rel_proj.project(abs_latent, anchors)
        outputs_rel.append(rel_latent)
    
    handle.remove()

    return {
        'nll': nll_sum / len(test_loader['test'].dataset),
        'loss': loss_sum / len(test_loader['test'].dataset),
        'accuracy': correct * 100.0 / len(test_loader['test'].dataset),
        'predictions': np.vstack(preds),
        'output_abs': abs_latent,
        'output_rel': torch.cat(outputs_rel, dim=1),
    }

def get_gradients(test_loader, model, criterion, layer_name: str, rel_proj, regularizer=None, **kwargs):

    pass



def predictions(test_loader, model, **kwargs):
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        input = input.cuda(non_blocking=True)
        output = model(input, **kwargs)
        probs = F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
           issubclass(module.__class__, curves._BatchNorm)


def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]


def update_bn(loader, model, **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0
    for input, _ in loader:
        input = input.cuda(non_blocking=True)
        batch_size = input.data.size(0)

        momentum = batch_size / (num_samples + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(input, **kwargs)
        num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))


def highlight_anchors(fig):
    for trace in fig.data:
        if trace.name == 'anchor':
            trace.marker.symbol = 'star'  # Set the marker type to 'star' for 'anchor'
            trace.marker.size = 5
            trace.marker.color = 'black' 
            trace.marker.line.width = .05  # Set the marker line width to 2 
        else:
            trace.marker.symbol = 'circle'
            trace.marker.line.width= .05


    for frame in fig.frames:
      for trace in frame.data:
        if trace.name == 'anchor':
            trace.marker.symbol = 'star'  # Set the marker type to 'star' for 'anchor'
            trace.marker.size = 5
            trace.marker.color = 'black'
            trace.marker.line.width = .05  # Set the marker line width to 2
        else:
            trace.marker.symbol = 'circle'
            trace.marker.line.width= .05
            
    return fig

class RelProjector(object):
    def __init__(self, proj_fn, center: bool = False, standardize: bool = False):
        self.proj_fn = proj_fn
        self.center = center
        self.standardize = standardize

    def project(self, input, anchors):
        if self.center is True:
            m = input.mean(dim=0, keepdim=True)
            input -= m
            anchors -= m
        if self.standardize is True:
            s = input.std(dim=0, keepdim=True)
            input /= s
            anchors /= s
        return self.proj_fn(input, anchors)

def cosine_projection(x, y):
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return torch.einsum('ik,jk->ij', x, y)

def basis_norm_projection(x, y):
    return torch.einsum('ik,jk->ij', x, y) / (y**2).sum(dim=1)

def euclidean_projection(x, y):
    return dist_projection(x, y, 2)

def dist_projection(x, y, p):
    y = y.permute(2, 0, 1)
    x = x.permute(2, 0, 1)
    return torch.cdist(x, y, p=p).permute(1, 2, 0)