# https://gist.github.com/omitakahiro/c49e5168d04438c5b20c921b928f1f5d

from scipy.sparse import linalg as splinalg
import scipy.sparse as sparse
import sys
import numpy as np

def sparse_cholesky(A): 
  
    n = A.shape[0]
    LU = splinalg.splu(A,diag_pivot_thresh=0) # sparse LU decomposition

    return LU.L.dot( sparse.diags(LU.U.diagonal()**0.5) )
