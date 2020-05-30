import torch
from .integrator import Integrator

class SymplecticEuler(Integrator):
    def __init__(self, x, m, h=0.033, dtype=torch.float32):
        super().__init__(x, m, h=h, dtype=dtype, x1=False)

    def step(self):
        if self.x.grad is not None:
            self.x.grad.zero_()
        h = self.h
        potential_energy = self.potential_energy(self.x)
        potential_energy.backward()
        force = -self.x.grad
        self.v.data.add_(self.m_inv.unsqueeze(-1) * force * h)
        self.x.data.add_(self.v * h)