import torch
import sim
import argparse
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.animation import FuncAnimation

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--integrator", type=str, default="symplectic-euler")
arg_parser.add_argument("--springs", type=str, default="hookean")
arg_parser.add_argument("--dtype", type=str, default="double")
arg_parser.add_argument("--optim-tol", type=float, default=1e-12)
arg_parser.add_argument("--k", type=float, default=8)
args = arg_parser.parse_args()

if args.dtype == "double":
    dtype = torch.double
elif args.dtype == "float":
    dtype = torch.float32
else:
    raise ValueError("invalid dtype: {}".format(args.dtype))

# vertex positions
x = torch.tensor([
    [0, 0],
    [1, 0],
    [2, 0.5]
], dtype=dtype)

# springs, specified as vertex indices
indices = [
    [0, 1],
    [1, 2],
    [2, 0]
]

# rest lengths
l0 = torch.tensor([
    1,
    1.3,
    1.4
], dtype=dtype)

num_vertices, d = x.shape
num_springs = len(indices)

# stiffness
k = args.k * torch.ones(num_springs, dtype=dtype)

# mass
m = torch.ones(num_vertices, dtype=dtype)

if args.integrator == "symplectic-euler":
    system = sim.integration.SymplecticEuler(x, m, dtype=dtype)
elif args.integrator == "avf4":
    system = sim.integration.AVF4(x, m, dtype=dtype, optim_tol=args.optim_tol)
elif args.integrator == "backward-euler":
    system = sim.integration.BackwardEuler(x, m, dtype=dtype, optim_tol=args.optim_tol)
else:
    raise ValueError("invalid integrator: {}".format(args.integrator))

system.set_springs(indices, l0, k, mode=args.springs)
    
fig, ax = plt.subplots()

def make_segment_data():
    segments = []
    x = system.x.detach()
    for ind in indices:
        a = x[ind[0]].tolist()
        b = x[ind[1]].tolist()
        segments.append([a, b])
    return segments

vertices = ax.scatter([], [])
segments = mc.LineCollection(make_segment_data())
text = ax.text(-0.9, 2.3, "")

def init():
    ax.add_collection(segments)
    ax.set_aspect(1)
    ax.set_xlim(-1, 3.5)
    ax.set_ylim(-2, 2.5)
    return vertices, segments, text

def update(frame):
    system.step()
    hamiltonian = system.hamiltonian()
    print("Hamiltonian: {}".format(hamiltonian))
    text.set_text("Hamiltonian: {0:.2f}".format(hamiltonian))

    segments.set_segments(make_segment_data())
    vertices.set_offsets(system.x.detach())
    return vertices, segments, text

anim = FuncAnimation(fig, update, frames=None, init_func=init, interval=30, blit=True)

plt.show()