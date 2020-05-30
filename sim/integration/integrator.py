import torch
import torch.nn as nn
from ..potentials import HookeanSprings
from ..potentials import StVKSprings

class Integrator(nn.Module):
    def __init__(self, x, m, h=0.033, dtype=torch.float32, x1=True):
        super().__init__()
        num_vertices, d = x.size()
        self.h = h

        self.register_buffer("x", x.detach().clone())
        if x1:
            self.register_buffer("x1", self.x.clone())
            self.x1.requires_grad_()
        else:
            self.x.requires_grad_()

        self.register_buffer("v", torch.zeros(num_vertices, d, dtype=torch.float32))
        self.register_buffer("m", m.detach().clone())
        self.register_buffer("m_inv", 1 / self.m)
        self.springs = None

    def set_springs(self, indices, l0, k, mode="hookean"):
        num_vertices = self.x.size()[0]
        if mode == "hookean":
            potential = HookeanSprings(indices, l0, k, num_vertices)
        elif mode == "stvk":
            potential = StVKSprings(indices, l0, k, num_vertices)
        else:
            raise ValueError("invalid spring mode: {}".format(mode))
        self.springs = potential

    def potential_energy(self, x=None):
        if x is None:
            x = self.x
        return self.springs.energy(x)

    def kinetic_energy(self):
        return 0.5 * (self.m.unsqueeze(-1) * self.v.pow(2)).sum()

    def hamiltonian(self):
        return self.potential_energy() + self.kinetic_energy()