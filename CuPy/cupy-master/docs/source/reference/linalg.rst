Linear Algebra
==============

.. https://docs.scipy.org/doc/numpy/reference/routines.linalg.html

Matrix and vector products
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   
   cupy.cross
   cupy.dot
   cupy.vdot
   cupy.inner
   cupy.outer
   cupy.matmul
   cupy.tensordot
   cupy.einsum
   cupy.linalg.matrix_power
   cupy.kron

Decompositions
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.linalg.cholesky
   cupy.linalg.qr
   cupy.linalg.svd

Matrix eigenvalues
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.linalg.eigh
   cupy.linalg.eigvalsh

Norms etc.
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.linalg.det
   cupy.linalg.norm
   cupy.linalg.matrix_rank
   cupy.linalg.slogdet
   cupy.trace


Solving linear equations
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.linalg.solve
   cupy.linalg.tensorsolve
   cupy.linalg.lstsq
   cupy.linalg.inv
   cupy.linalg.pinv
   cupy.linalg.tensorinv

   cupyx.scipy.linalg.lu_factor
   cupyx.scipy.linalg.lu_solve
   cupyx.scipy.linalg.solve_triangular
