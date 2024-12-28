import dolfinx
from dolfinx import fem
from basix.ufl import element, mixed_element
from dolfinx import plot, mesh
from mpi4py import MPI
import pyvista 
from dolfinx.fem import Constant, Function, functionspace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx.io import VTXWriter
from dolfinx.mesh import create_unit_square
from dolfinx.plot import vtk_mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)

from scipy_converter import *
import matplotlib.pyplot as plt

import numpy as np
from vortex_sw import *
import ufl
from ufl import (FacetNormal, Identity, TestFunction, TrialFunction, TestFunctions, TrialFunctions,
                 div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym)
from petsc4py import PETSc

T_end = 0.5
max_iter=100000


domain = mesh.create_unit_square(MPI.COMM_WORLD,50, 50)#, mesh.CellType.quadrilateral)

# Plot the mesh
tdim = domain.topology.dim

domain.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, show_edges=True)
# plotter.view_xy()
# plotter.show()


v_cg1 = element("CG", domain.topology.cell_name(), 1, shape=(domain.geometry.dim, ))
s_cg1 = element("CG", domain.topology.cell_name(), 1)

w_cg11 = mixed_element([s_cg1, v_cg1])

W = functionspace(domain, w_cg11)

Q,_ = W.sub(0).collapse()

(h, u) = TrialFunctions(W)
(q, v) = TestFunctions(W)

trial = TrialFunction(W)
test = TrialFunction(W)

wn = Function(W)
wn1 = Function(W)
hn, un  = wn.split()
hn1,un1 = wn1.split()
w_plot = Function(W)
h_plot = Function(Q)
#,u_plot = wn1.split()

# Defining my dofs
wn.x.array[:] = np.arange(len(wn.x.array))
h_plot.interpolate(hn)
h_dofs = np.int64(h_plot.x.array.copy())
h_plot.interpolate(un.sub(0))
u_dofs = np.int64(h_plot.x.array.copy())
h_plot.interpolate(un.sub(1))
v_dofs = np.int64(h_plot.x.array.copy())


un.sub(0).interpolate(lambda x : analytic_travelling_vortexU(x[0],x[1],0.5,0.5,t=0))
un.sub(1).interpolate(lambda x : analytic_travelling_vortexV(x[0],x[1],0.5,0.5,t=0))
hn.interpolate(lambda x: 1*(x[0]<10000.))# Constant(domain,1.)##.interpolate(lambda x : analytic_travelling_vortexH(x[0],x[1],0.5,0.5,t=0))
# hn.interpolate(lambda x: 1 + 0.1*(x[0]<0.6)*(x[0]>0.4)*(x[1]<0.2)*(x[1]>0.1))# Constant(domain,1.)##.interpolate(lambda x : analytic_travelling_vortexH(x[0],x[1],0.5,0.5,t=0))


w0 = wn.copy()

# Plot IC
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(Q)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
h_plot.interpolate(hn)
u_grid.point_data["h"] = h_plot.x.array.real
u_grid.set_active_scalars("h")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
u_plotter.show()


u_topology, u_cell_types, u_geometry = plot.vtk_mesh(Q)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
h_plot.interpolate(un.sub(0))
u_grid.point_data["u"] = h_plot.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
u_plotter.show()

u_topology, u_cell_types, u_geometry = plot.vtk_mesh(Q)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
h_plot.interpolate(un.sub(1))
u_grid.point_data["v"] = h_plot.x.array.real
u_grid.set_active_scalars("v")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
u_plotter.show()

h_mesh = ufl.CellDiameter(domain)
num_cells = domain.topology.index_map(tdim).size_local
minh = np.min(domain.h(2, np.arange(num_cells)))

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

# Method
print("minumum h ", minh)
alpha = 0.5
dt_np = 0.1*minh/np.sqrt(2)/1.    
dt = Constant(domain, dt_np)
res = (inner(v+alpha*h_mesh*grad(q),u-un + dt*grad(hn) )\
        +dot(q+alpha*h_mesh*div(v), h-hn+dt*div(un)))*dx


# res = (inner(v,u)+dot(q,h))*dx + (div(v)+q)*dx

lhs_part = form(lhs(res))
rhs_part = form(rhs(res))


boundary_dofs = fem.locate_dofs_topological(W, fdim, boundary_facets)
bc = fem.dirichletbc(w0, boundary_dofs)

A = assemble_matrix(lhs_part, bcs = [bc])
A.assemble()

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

A_Scipy = PETSc2scipy(A)

# plt.spy(A_Scipy)
# plt.show()

# for i in range(A_Scipy.shape[0]):
#     if A_Scipy[i,i]==0:
#         print(f"Index {i} is zero")



t =0.
it =0
while (t<T_end and it < max_iter):
    print(f"Time {t}, it {it}, dt {dt_np}", end ="\r")
    # dt_np = 0.5*minh/1.    
    # dt_fake = Constant(domain, dt_np) # to be fixed
    t+=dt_np
    it+=1
    
    b = assemble_vector(rhs_part)
    apply_lifting(b, [lhs_part],[[bc]])
    
    solver.solve(b, wn1.x.petsc_vec)
    wn1.x.scatter_forward()
    
    # Brute force BC
    wn1.x.array[boundary_dofs] = w0.x.array[boundary_dofs]
    
    wn.x.array[:] = wn1.x.array
    

    # u_topology, u_cell_types, u_geometry = plot.vtk_mesh(Q)
    # u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    # h_plot.interpolate(hn)#.sub(0))
    # u_grid.point_data["u"] = h_plot.x.array.real
    # u_grid.set_active_scalars("u")
    # u_plotter = pyvista.Plotter()
    # u_plotter.add_mesh(u_grid, show_edges=True)
    # u_plotter.view_xy()
    # u_plotter.show()
        

# Plot IC
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(Q)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
h_plot.interpolate(hn)
u_grid.point_data["h"] = h_plot.x.array.real
u_grid.set_active_scalars("h")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
u_plotter.show()


u_topology, u_cell_types, u_geometry = plot.vtk_mesh(Q)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
h_plot.interpolate(un.sub(0))
u_grid.point_data["u"] = h_plot.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
u_plotter.show()

u_topology, u_cell_types, u_geometry = plot.vtk_mesh(Q)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
h_plot.interpolate(un.sub(1))
u_grid.point_data["v"] = h_plot.x.array.real
u_grid.set_active_scalars("v")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
u_plotter.show()



u_topology, u_cell_types, u_geometry = plot.vtk_mesh(Q)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
h_plot.interpolate(un.sub(1))
u_grid.point_data["vel2"] = h_plot.x.array.real**2
h_plot.interpolate(un.sub(0))
u_grid.point_data["vel2"] += h_plot.x.array.real**2
u_grid.set_active_scalars("vel2")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
u_plotter.show()