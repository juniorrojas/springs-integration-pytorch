import torch

def make_incidence(indices, num_vertices, dtype=torch.float32):
    # this creates a dense matrix (incidence), but
    # sparse matrices or convolutions might be more appropriate
    # in certain cases
    num_springs = len(indices)
    incidence = torch.zeros(num_springs, num_vertices, dtype=dtype)
    for i, item in enumerate(indices):
        i1, i2 = item
        incidence[i, i1] = 1
        incidence[i, i2] = -1
    return incidence