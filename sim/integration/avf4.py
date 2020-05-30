import torch
from .integrator import Integrator
from .optim import GradientDescentWithLineSearch

class AVF4(Integrator):
    def __init__(self, x, m, h=0.033, dtype=torch.float32,
                 optim_tol=1e-8, step_size_eps=1e-20,
                 max_line_search_iters=500):
        super().__init__(x, m, h=h, dtype=dtype)
        self.optimizer = GradientDescentWithLineSearch(
            self.x1, max_line_search_iters,
            optim_tol, step_size_eps, self.loss)

    def loss(self, x):
        h = self.h
        y = self.y
        return h * h * self.potential_energy(x) + h * h * 8 * self.potential_energy((x + self.x) * 0.5) + 6 * (x - y).pow(2).sum()

    def step(self):
        h = self.h
        self.x1.data.copy_(self.x)
        e = self.potential_energy(self.x1)
        e.backward()
        self.f0 = -self.x1.grad.detach().clone()
        self.x1.grad.zero_()
        self.y = self.x + self.v * h + h * h / 12 * self.f0
        self.x1.data.copy_(self.x + self.v * h)
        
        self.optimizer.step()
        
        self.v.data.copy_(2 * (self.x1 - self.x) / h - self.v)
        self.x.data.copy_(self.x1)