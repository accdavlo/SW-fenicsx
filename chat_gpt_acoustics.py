import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io
from ufl import TrialFunction, TestFunction, dx, grad, dot, inner, split, CellDiameter
from ufl import rhs, lhs, form
from petsc4py import PETSc

from vortex_sw import *


# Domain and mesh
L = 1.0  # Length of the square domain
N = 10   # Number of elements per side
mesh_square = mesh.create_rectangle(
    MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([L, L])], [N, N], cell_type=mesh.CellType.triangle
)

# Function space
V = fem.functionspace(mesh_square, ("CG", 1))

# Trial and Test functions
u, v, p = TrialFunction(V), TrialFunction(V), TrialFunction(V)
phi = TestFunction(V)

# Time step parameters
dt = 0.01  # Time step size
T = 10.0    # Final time

def vortex_u(x, y, x0, y0, t):
    # Define traveling vortex U function
    return analytic_travelling_vortexU(x, y, x0, y0, t)

def vortex_v(x, y, x0, y0, t):
    # Define traveling vortex V function
    return analytic_travelling_vortexV(x, y, x0, y0, t)

# Initial conditions
x0, y0 = L / 2, L / 2  # Center of vortex
u_0 = fem.Function(V)
v_0 = fem.Function(V)
p_0 = fem.Function(V)

# Initialize the initial conditions
coords = mesh_square.geometry.x
x, y = coords[:, 0], coords[:, 1]

u_0.x.array[:] = vortex_u(x, y, x0, y0, 0.0)
v_0.x.array[:] = vortex_v(x, y, x0, y0, 0.0)
p_0.x.array[:] = 1.0

# Define variational problem
u_n, v_n, p_n = u_0.copy(), v_0.copy(), p_0.copy()
nu, nv, np = fem.TrialFunction(V), fem.TrialFunction(V), fem.TrialFunction(V)


tdim = mesh.topology.dim
h = CellDiameter(mesh)
num_cells = mesh.topology.index_map(tdim).size_local
minh = np.min(mesh.h(2, np.arange(num_cells)))

alpha = 0.1  # Stabilization parameter

# Variational formulation
a1 = phi * (nu - u_n) / dt * dx + phi * (grad(p_n)[0]) * dx
supg_u = alpha *h* dot(grad(phi)[0], grad(p_n)[0] + grad(u_n)[0] + grad(v_n)[1]) * dx
L1 = rhs(a1+supg_u)

a2 = phi * (nv - v_n) / dt * dx + phi * (grad(p_n)[1]) * dx
supg_v = alpha *h* dot(grad(phi)[1], grad(p_n)[1]) * (grad(p_n)[0] + grad(u_n)[0] + grad(v_n)[1]) * dx
L2 = rhs(a2+supg_v)

a3 = phi * (np - p_n) / dt * dx + phi * (grad(u_n)[0] + grad(v_n)[1]) * dx
supg_p = (
    alpha*h * dot(grad(phi), grad(u_n)[0]) * (grad(u_n)[0] + grad(p_n)[0]) * dx +
    alpha*h * dot(grad(phi), grad(v_n)[1]) * (grad(v_n)[1] + grad(p_n)[1]) * dx
)
L3 = -phi * (grad(u_n)[0] + grad(v_n)[1]) * dx

# Assemble variational problems
a_u = a1 + supg_u
L_u = L1
a_v = a2 + supg_v
L_v = L2
a_p = a3 + supg_p
L_p = L3

# Time-stepping loop
u_k, v_k, p_k = fem.Function(V), fem.Function(V), fem.Function(V)
nu_k, nv_k, np_k = fem.Function(V), fem.Function(V), fem.Function(V)
time = 0.0
while time < T:
    time += dt

    # Solve for u
    problem_u = fem.petsc.LinearProblem(a_u, L_u, bcs=[])
    nu_k.x.array[:] = problem_u.solve().x.array[:]

    # Solve for v
    problem_v = fem.petsc.LinearProblem(a_v, L_v, bcs=[])
    nv_k.x.array[:] = problem_v.solve().x.array[:]

    # Solve for p
    problem_p = fem.petsc.LinearProblem(a_p, L_p, bcs=[])
    np_k.x.array[:] = problem_p.solve().x.array[:]

    # Update fields for next step
    u_n.x.array[:] = nu_k.x.array[:]
    v_n.x.array[:] = nv_k.x.array[:]
    p_n.x.array[:] = np_k.x.array[:]

# Output results
with io.XDMFFile(mesh_square.comm, "acoustic_results.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh_square)
    xdmf.write_function(nu_k, time)
    xdmf.write_function(nv_k, time)
    xdmf.write_function(np_k, time)
