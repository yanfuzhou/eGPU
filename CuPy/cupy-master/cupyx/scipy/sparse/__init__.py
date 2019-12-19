from cupyx.scipy.sparse.base import issparse  # NOQA
from cupyx.scipy.sparse.base import isspmatrix  # NOQA
from cupyx.scipy.sparse.base import spmatrix  # NOQA
from cupyx.scipy.sparse.coo import coo_matrix  # NOQA
from cupyx.scipy.sparse.coo import isspmatrix_coo  # NOQA
from cupyx.scipy.sparse.csc import csc_matrix  # NOQA
from cupyx.scipy.sparse.csc import isspmatrix_csc  # NOQA
from cupyx.scipy.sparse.csr import csr_matrix  # NOQA
from cupyx.scipy.sparse.csr import isspmatrix_csr  # NOQA
from cupyx.scipy.sparse.dia import dia_matrix  # NOQA
from cupyx.scipy.sparse.dia import isspmatrix_dia  # NOQA

from cupyx.scipy.sparse.construct import eye  # NOQA
from cupyx.scipy.sparse.construct import identity  # NOQA
from cupyx.scipy.sparse.construct import rand  # NOQA
from cupyx.scipy.sparse.construct import random  # NOQA
from cupyx.scipy.sparse.construct import spdiags  # NOQA
from cupyx.scipy.sparse.construct import diags  # NOQA

# TODO(unno): implement bsr_matrix
# TODO(unno): implement dok_matrix
# TODO(unno): implement lil_matrix

# TODO(unno): implement kron
# TODO(unno): implement kronsum
# TODO(unno): implement diags
# TODO(unno): implement block_diag
# TODO(unno): implement tril
# TODO(unno): implement triu
# TODO(unno): implement bmat
# TODO(unno): implement hstack
# TODO(unno): implement vstack

# TODO(unno): implement save_npz
# TODO(unno): implement load_npz

# TODO(unno): implement find

# TODO(unno): implement isspmatrix_bsr(x)
# TODO(unno): implement isspmatrix_lil(x)
# TODO(unno): implement isspmatrix_dok(x)

from cupyx.scipy.sparse import linalg  # NOQA