# WIP toward hackathon kernel

import numpy as np
import loopy as lp
from scipy import sparse
lp.set_caching_enabled(False)
from tvb_hpc import model, coupling, network, utils, compiler, scheme
from tvb_hpc.numba import NumbaTarget

LOG = utils.getLogger('hackathon')

target = NumbaTarget()

osc = model.Kuramoto()
osc.dt = 1.0
osc_knl = osc.kernel(target)

cfun = coupling.Kuramoto(osc)
net = network.Network(osc, cfun)
net_knl = net.kernel(target)

scm = scheme.EulerStep(osc.dt)
scm_knl = scm.kernel(target)
scm_knl = lp.fix_parameters(scm_knl, nsvar=len(osc.state_sym))

knls = osc_knl, net_knl, scm_knl
data_flow = [
    ('input', 1, 0),
    ('diffs', 0, 2),
    ('drift', 0, 2),
]
knl = lp.fuse_kernels(knls, data_flow=data_flow)


knl = lp.to_batched(knl, 'nstep', [], 'i_step', sequential=True)

import pymbolic as pm
knl = lp.fix_parameters(knl, i_time=pm.parse('i_step % ntime'))

# TODO next -> state, i_step -> i_time

target.get_kernel_executor(knl)

# load connectivity
npz = np.load('data/hcp0.npz')
weights = npz['weights']
lengths = npz['lengths']
weights /= weights.max()
nnode = weights.shape[0]
nz = ~(weights == 0)
nnz = nz.sum()
wnz = weights[nz]
lnz = (lengths[nz] / osc.dt).astype(np.uintc)
sw = sparse.csr_matrix(weights)
col = sw.indices.astype(np.uintc)
row = sw.indptr.astype(np.uintc)

# build other data arrays
state, input, param, drift, diffs, _ = osc.prep_arrays(nnode)
obsrv = np.zeros((lnz.max() + 3, nnode, 1), np.float32)
LOG.info('obsrv %r %.3f MB', obsrv.shape, obsrv.nbytes / 2**20)

# nstep, nnode, ntime, state, input, param, drift, diffs, obsrv, nnz, delays, row, col, weights
trace = np.zeros((400, ) + state.shape, np.float32)
import time
tic = time.time()
for i in range(400):
    knl(100, nnode, obsrv.shape[0], state, input, param, drift, diffs, obsrv, nnz, lnz, row, col, wnz)
    trace[i] = state

toc = time.time()
print(toc - tic, 's elapsed')
