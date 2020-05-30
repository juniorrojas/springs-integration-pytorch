import torch
from .integrator import Integrator
from .optim import GradientDescentWithLineSearch

class BackwardEuler(Integrator):
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
        return h * h * self.potential_energy(x) + 0.5 * (self.m * (x - y).pow(2).sum(1)).sum()

    def step(self):
        h = self.h
        self.y = self.x + self.v * h
        self.x1.data.copy_(self.x + self.v * h)
        self.optimizer.step()
        self.v.data.copy_((self.x1 - self.x) / h)
        self.x.data.copy_(self.x1)