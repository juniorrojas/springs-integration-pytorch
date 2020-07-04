import torch
import torch.nn as nn
from . import common

class HookeanSprings(nn.Module):
    def __init__(self, indices, l0, k, num_vertices):
        super().__init__()
        self.indices = indices
        assert l0.shape[0] == k.shape[0]
        self.register_buffer("incidence", common.make_incidence(indices, num_vertices, l0.dtype))
        self.register_buffer("l0", l0)
        self.register_buffer("k", k)

    def energy(self, x):
        d = self.incidence.mm(x)
        q = d.pow(2).sum(1)
        l = (q + 1e-6).sqrt()
        dl = l - self.l0
        e = 0.5 * (self.k * dl.pow(2)).sum()
        return e