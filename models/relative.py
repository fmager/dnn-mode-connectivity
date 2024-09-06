import torch
import torch.nn as nn
import torch.nn.functional as F


class RelCos(nn.Module):
    def __init__(self, anchors):
        super(RelCos, self).__init__()
        self.anchors = nn.Parameter(F.normalize(anchors, p=2, dim=1), requires_grad=False)

    def forward(self, x, n_anchors):
        '''
        x: (N, in_features, samples)
        anchors: (M, in_features, samples)
        '''
        anchors = self.anchors[:n_anchors, :, :]
        x = F.normalize(x, p=2, dim=1)
        return torch.einsum('nik,mik->nmk', x, anchors)
    
class RelBasis(nn.Module):
    def __init__(self, anchors, norm: bool = False):
        super(RelBasis, self).__init__()

        self.anchors = nn.Parameter(anchors, requires_grad=False)
        self.norm = norm

    def forward(self, x, n_anchors):
        '''
        x: (N, in_features, samples)
        anchors: (M, in_features, samples)
        '''
        anchors = self.anchors[:n_anchors, :, :]
        out = torch.einsum('nik,mik->nmk', x, anchors)
        if self.norm:
            out /= (anchors**2).sum(dim=1)
        return out
        
class RelDist(nn.Module):
    def __init__(self, anchors, p: int = 2):
        super(RelDist, self).__init__()

        self.anchors = nn.Parameter(anchors, requires_grad=False)
        self.p = p

    def forward(self, x, n_anchors):
        '''
        x: (N, in_features, samples)
        anchors: (M, in_features, samples)
        '''
        anchors = self.anchors[:n_anchors, :, :]
        anchors = anchors.permute(2, 0, 1)
        x = x.permute(2, 0, 1) # (samples, N, in_features)
        return torch.cdist(x, anchors, p=self.p).permute(1, 2, 0)
    
