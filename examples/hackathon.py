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
osc.dt = 0.1
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

knl = lp.to_batched(knl, 'nstep', [], 'i_time', sequential=True)

# TODO next -> state, i_step -> i_time

target.get_kernel_executor(knl)

# load connectivity
npz = np.load('data/hcp0.npz')
weights = npz['weights']
lengths = npz['lengths']
weights /= weights.max()
nnode = weights.shape[0]
nnz = ~(weights == 0)
wnz = weights[nnz]
lnz = (lengths[nnz] / osc.dt).astype(np.uintc)

# build other data arrays
next, state, drift = np.zeros((3, nnode, 2), np.float32)
input, param, diffs = np.zeros((3, nnode, 2), np.float32)
obsrv = np.zeros((lnz.max() + 3, nnode, 2), np.float32)
LOG.info('obsrv %r %.3f MB', obsrv.shape, obsrv.nbytes / 2**20)

# nstep, nnode, i_time, ntime, state, input, param, drift, diffs, obsrv, nnz, delays, row, col, weights

