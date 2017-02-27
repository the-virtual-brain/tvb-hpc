
"""
Run single simulation w/ 515k connectivity.

"""

from typing import Tuple
import loopy as lp
lp.set_caching_enabled(False)
import numpy as np
from scipy import io, sparse
from tvb_hpc import model, coupling, network, utils, compiler, scheme


LOG = utils.getLogger('run515k')

# load data
nn = 515056
data = io.loadmat('data/conn515k.mat')
def from_matlab(key: str) -> sparse.csr_matrix:
    _ = lambda a: a[0, 0][0]
    mat = data[key]
    return sparse.csc_matrix(
        ( _(mat['data']), _(mat['ir']), _(mat['jc']) ),
        shape=(nn, nn),
    ).tocsr()
W = from_matlab('Mat')
L = from_matlab('FL')
W = W.multiply(L > 0)
assert L.shape == W.shape == (nn, nn)
assert L.nnz == W.nnz
assert L.has_sorted_indices
assert W.has_sorted_indices
pct = 100 * L.nnz / (nn * nn)
LOG.info('shape %r, nnz %r %.4f %%', L.shape, L.nnz, pct)



# TODO make compiler instance reusable/refactor build
# TODO ispc usuable on macos?
def comp():
    comp = compiler.Compiler(cc='/usr/local/bin/gcc-6')
    comp.cflags += '-fopenmp -march=native -ffast-math'.split()
    comp.ldflags += '-fopenmp'.split()
    return comp


# build kernels
target = compiler.OpenMPCTarget()

osc = model.G2DO()
osc.dt = 0.1
osc_knl = osc.kernel(target)
osc_fn = compiler.CompiledKernel(osc_knl, comp())

cfun = coupling.Linear(osc)
net = network.Network(osc, cfun)
net_knl = net.kernel(target)
net_fn = compiler.CompiledKernel(net_knl, comp())

scm = scheme.EulerStep(osc.dt)
scm_knl = scm.kernel(target)
scm_knl = lp.prioritize_loops(scm_knl, ['i', 'j'])
scm_knl = lp.fix_parameters(scm_knl, nsvar=2)
scm_knl = lp.tag_inames(scm_knl, [('j', 'ilp')])
scm_fn = compiler.CompiledKernel(scm_knl, comp())


# prepare data
L_data: np.ndarray = L.data
D = (L_data / 1.0).astype(np.uint32)
Dmax = D.max()
nnode = W.shape[0]

next, state, drift = np.zeros((3, nnode, 2), np.float32)
input, param, diffs = np.zeros((3, nnode, 2), np.float32)
obsrv = np.zeros((Dmax + 3, nnode, 2), np.float32)
LOG.info('obsrv %r %.3f MB', obsrv.shape, obsrv.nbytes / 2**20)


def step(n_step=1):
    for _ in range(n_step):
        t = Dmax + 1
        net_fn(t=t, ntime=Dmax + 3, nnode=nnode, nnz=W.nnz,
               row=W.indptr, col=W.indices,
               delays=D, weights=W.data,
               input=input, obsrv=obsrv
               )
        osc_fn(nnode=nnode,
               state=state, input=input, param=param,
               drift=drift, diffs=diffs, obsrv=obsrv[t])
        scm_fn(nnode=nnode, nsvar=2, next=next, state=state, drift=drift)


# warm up
step(2)

# time it
with utils.timer('10 time steps'):
    step(10)
