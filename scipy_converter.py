import petsc4py
from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np

def eliminate_zeros(sparse_M):
    # Input: scipy sparse matrix, remove all data smaller than 3e-14 in abs value
    sparse_M.data[np.abs(sparse_M.data)<3e-14] = 0.
    sparse_M.eliminate_zeros()
    return sparse_M

def PETSc2scipy(M):
    # Takes in input dolfin.cpp.la.Matrix and returns the csr_matrix in scipy
    s1 = M.size[0]
    s2 = M.size[1]
    (ai,aj,av) = M.getValuesCSR()
    sparse_M = sparse.csr_matrix((av,aj,ai), shape= [s1,s2] )
    sparse_M = eliminate_zeros(sparse_M)
    return sparse_M