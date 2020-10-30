import torch
import torch.optim as optim

def line_search(x, descent_dir, loss, eval_loss,
                max_line_search_iters, step_size_eps,
                k=0.3):
    step_size = 1.0
    x1 = x + descent_dir * step_size
    loss1 = eval_loss(x1)
    iters = 0
    while loss1 >= loss and iters < max_line_search_iters:
        step_size *= k
        if step_size < step_size_eps:
            break
        x1 = x + descent_dir * step_size
        loss1 = eval_loss(x1)
        iters += 1
    return step_size

class GradientDescentWithLineSearch:
    def __init__(self, x,
                 max_line_search_iters,
                 grad_eps, step_size_eps,
                 eval_loss, max_iters=5000):
        self.max_line_search_iters = max_line_search_iters
        self.max_iters = max_iters
        self.eval_loss = eval_loss
        self.x = x
        self.max_line_search_iters = max_line_search_iters
        assert isinstance(step_size_eps, float)
        self.step_size_eps = step_size_eps
        self.grad_eps2 = grad_eps * grad_eps

    def step(self):
        if self.x.grad is not None:
            self.x.grad.zero_()
        for _ in range(self.max_iters):
            loss = self.eval_loss(self.x)
            loss.backward()
            error2 = self.x.grad.pow(2).sum()
            if error2 < self.grad_eps2:
                break
            descent_dir = -self.x.grad
            step_size = line_search(
                self.x,
                descent_dir,
                loss=loss,
                eval_loss=self.eval_loss,
                max_line_search_iters=self.max_line_search_iters,
                step_size_eps=self.step_size_eps)
            if step_size < self.step_size_eps:
                break
            self.x.data.add_(descent_dir * step_size)
            self.x.grad.zero_()