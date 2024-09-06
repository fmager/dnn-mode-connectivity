import torch

def sample_anchors(X, procedure: str = 'random', n_anchors: int = 768):
    """
    X: torch.Tensor of shape (n_samples, n_features)
    n_anchors: int, number of anchors to sample
    procedure: str, type of anchor sampler
    range: tuple, min and max length for the anchor vectors
    """
    if procedure == 'rand':
        idx = torch.randperm(X.size(0))[:n_anchors]
    elif procedure == 'furth_sum':
        idx = furth_sum(X, n_anchors)
    elif procedure == 'furth':
        idx = furth(X, n_anchors)
    elif procedure == 'furth_cos':
        idx = furth_cos(X, n_anchors)
    elif procedure == 'furth_sum_cos':
        idx = furth_sum_cos(X, n_anchors)
    else:
        raise ValueError(f"Invalid procedure: {procedure}")
    
    # check if the anchors are unique
    if len(set(idx.tolist())) != len(idx):
        raise ValueError(f'Anchors of procedure {procedure} are not unique. Found {len(set(idx.tolist()))} unique anchors out of {len(idx)}')
    
    return idx

def furth(X, n_anchors):

    j = torch.randint(0, X.size(0), (1,)) # random first, not added to set
    q = torch.norm(X - X[j], dim=1) # distances

    j = torch.argmax(q) # new anchor
    C = j.unsqueeze(0) # add to set
    q = torch.norm(X - X[j], dim=1) # distances
    q[j] = -float("Inf") # set distance to negative infinity


    # iterate
    for _ in range(1, n_anchors):
        j = torch.argmax(q) # new anchor
        # if j in C:
        #     raise ValueError(f'Anchor {j} already in set')
        q[j] = -float("Inf") # set distance to negative infinity

        C = torch.cat((C, j.unsqueeze(0)), dim=0) # add to set
        q = torch.min(q, torch.norm(X - X[j], dim=1)) # update distances
    
    return C

def furth_sum(X, n_anchors):

    j = torch.randint(0, X.size(0), (1,)) # random first, not added to set
    q = torch.norm(X - X[j], dim=1) # distances

    j = torch.argmax(q) # new anchor
    C = j.unsqueeze(0) # add to set
    q = torch.norm(X - X[j], dim=1) # distances
    q[j] = -float("Inf") # set distance to 0

    for _ in range(1, n_anchors):
        j = torch.argmax(q) # new anchor
        # if j in C:
        #     raise ValueError(f'Anchor {j} already in set')
        q[j] = -float("Inf") # set distance to negative infinity
        C = torch.cat((C, j.unsqueeze(0)), dim=0) # add to set
        q += torch.norm(X - X[j], dim=1)

    return C
     
def furth_cos(X, n_anchors):

    X_ = X - X.mean(dim=0, keepdim=True)
    X_ = X_ / X_.norm(dim=1, keepdim=True)

    j = torch.randint(0, X.size(0), (1,)) # random first, not added to set
    q = torch.norm(X - X[j], dim=1) # distances

    j = torch.argmax(q) # new anchor
    C = j.unsqueeze(0) # add to set
    w = X_ @ X_[j].t().squeeze() # cosine similarity
    q = torch.norm(X - X[j], dim=1) / (torch.abs(w) + 1e-6) # distances
    q[j] = -float("Inf") # set distance to negative infinity


    for _ in range(1, n_anchors):
        j = torch.argmax(q) # new anchor
        q[j] = -float("Inf") # set distance to negative infinity
        C = torch.cat((C, j.unsqueeze(0)), dim=0) # add to set
        w = X_ @ X_[j].t().squeeze()
        q = torch.min(q, torch.norm(X - X[j], dim=1)) / (torch.abs(w) + 1e-6) # update distances

    return C

def furth_sum_cos(X, n_anchors):

    X_ = X - X.mean(dim=0, keepdim=True)
    X_ = X_ / X_.norm(dim=1, keepdim=True)

    j = torch.randint(0, X.size(0), (1,)) # random first, not added to set
    q = torch.norm(X - X[j], dim=1) # distances

    j = torch.argmax(q) # new anchor
    C = j.unsqueeze(0) # add to set
    w = X_ @ X_[j].t().squeeze() # cosine similarity
    q = torch.norm(X - X[j], dim=1) / (torch.abs(w) + 1e-6) # distances
    q[j] = -float("Inf") # set distance to negative infinity

    for _ in range(1, n_anchors):
        j = torch.argmax(q) # new anchor
        q[j] = -float("Inf") # set distance to negative infinity
        C = torch.cat((C, j.unsqueeze(0)), dim=0) # add to set
        w = X_ @ X_[j].t().squeeze()
        q += torch.norm(X - X[j], dim=1) / (torch.abs(w) + 1e-6) # update distances

    return C