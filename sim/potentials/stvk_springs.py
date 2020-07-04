import torch
import torch.nn as nn
from . import common

class StVKSprings(nn.Module):
    def __init__(self, indices, l0, k, num_vertices):
        super().__init__()
        self.indices = indices
        assert l0.shape[0] == k.shape[0]
        self.register_buffer("incidence", common.make_incidence(indices, num_vertices, dtype=l0.dtype))
        self.register_buffer("q0", l0.pow(2))
        self.register_buffer("k", k)

    def energy(self, x):
        d = self.incidence.mm(x)
        q = d.pow(2).sum(1)
        dq = q - self.q0
        e = 0.5 * (self.k * dq.pow(2)).sum()
        return e