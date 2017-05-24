# WIP toward hackathon kernel

import itertools
import numpy as np
import loopy as lp
import pymbolic as pm
from scipy import sparse
lp.set_caching_enabled(False)
from tvb_hpc import model, coupling, network, utils, compiler, scheme
from tvb_hpc.numba import NumbaTarget

LOG = utils.getLogger('hackathon')

target = NumbaTarget()

# build individual kernels
osc = model.Kuramoto()
osc.dt = 1.0
osc.const['omega'] = 10.0 * 2.0 * np.pi / 1e3
osc_knl = osc.kernel(target)

cfun = coupling.Kuramoto(osc)
cfun.param['a'] = pm.parse('a')
net = network.Network(osc, cfun)
net_knl = net.kernel(target)

scm = scheme.EulerStep(osc.dt)
scm_knl = scm.kernel(target)
scm_knl = lp.fix_parameters(scm_knl, nsvar=len(osc.state_sym))

# fuse kernels
knls = osc_knl, net_knl, scm_knl
data_flow = [ ('input', 1, 0), ('diffs', 0, 2), ('drift', 0, 2), ]
knl = lp.fuse_kernels(knls, data_flow=data_flow)

# and time step
knl = lp.to_batched(knl, 'nstep', [], 'i_step', sequential=True)
knl = lp.fix_parameters(knl, i_time=pm.parse('(i_step + i_step_0) % ntime'))
knl.args.append(lp.ValueArg('i_step_0', np.uintc))
knl = lp.add_dtypes(knl, {'i_step_0': np.uintc})

# TODO add outer time loop & prange over subjects?

# load connectivity TODO util function / class
npz = np.load('data/hcp0.npz')
weights = npz['weights']
lengths = npz['lengths']
weights /= weights.max()
nnode = weights.shape[0]
nz = ~(weights == 0)
nnz = nz.sum()
wnz = weights[nz]
sw = sparse.csr_matrix(weights)
col = sw.indices.astype(np.uintc)
row = sw.indptr.astype(np.uintc)

# kernel
import numpy as _lpy_np
import numba as _lpy_numba

@_lpy_numba.jit
def knl3(nstep, nnode, ntime, state, input, param, drift, diffs, obsrv, nnz, delays, row, col, weights, a, i_step_0):
    pi2 = _lpy_numba.float32(2*np.pi)
    for i_step in range(0, -1 + nstep + 1):
        for i_node in range(0, -1 + nnode + 1):
            j_node_lo = row[i_node]
            diffs[i_node, 0] = 0
            acc_j_node = 0
            omega = param[i_node, 0]
            j_node_hi = row[i_node + 1]
            for j_node in range(j_node_lo, -1 + j_node_hi + 1):
                acc_j_node = acc_j_node + weights[j_node]*_lpy_np.sin(obsrv[((i_step + i_step_0) % ntime) + -1*delays[j_node], col[j_node], 0] + -1*obsrv[((i_step + i_step_0) % ntime), i_node, 0])
            input[i_node, 0] = a*acc_j_node / nnode
            I = input[i_node, 0]
            drift[i_node, 0] = omega + I
            i_svar = 0

            state[i_node, i_svar] = state[i_node, i_svar] + drift[i_node, i_svar]
            if state[i_node, i_svar] > pi2:
                state[i_node, i_svar] -= pi2
            if state[i_node, i_svar] < 0.0:
                state[i_node, i_svar] += pi2
            theta = state[i_node, 0]
            obsrv[((i_step + i_step_0) % ntime), i_node, 1] = _lpy_np.sin(theta)
            obsrv[((i_step + i_step_0) % ntime), i_node, 0] = theta

# choose param space
nc, ns = 32, 32
couplings = np.logspace(0, 1.0, nc)
speeds = np.logspace(0.0, 2.0, ns)
trace = np.zeros((nc * ns, 400) + (nnode, ), np.float32)
LOG.info('trace.nbytes %.3f MB', trace.nbytes / 2**20)
for j, (speed, coupling) in enumerate(itertools.product(speeds, couplings)):
    lnz = (lengths[nz] / speed / osc.dt).astype(np.uintc)
    state, input, param, drift, diffs, _ = osc.prep_arrays(nnode)
    obsrv = np.zeros((lnz.max() + 3 + 4000, nnode, 2), np.float32)
    for i in range(trace.shape[1]):
        knl3(10, nnode, obsrv.shape[0], state, input, param,
                drift, diffs, obsrv, nnz, lnz, row, col, wnz,
                a=coupling, i_step_0=i*10)
        trace[j, i] = obsrv[i*10:(i+1)*10, :, 0].sum(axis=0)
    print(j)

# check correctness
n_work_items = nc * ns
r, c = np.triu_indices(nnode, 1)
win_size = 200 # 2s
tavg = np.transpose(trace, (1, 2, 0))
win_tavg = tavg.reshape((-1, win_size) + tavg.shape[1:])
err = np.zeros((len(win_tavg), n_work_items))
for i, tavg_ in enumerate(win_tavg):
    for j in range(n_work_items):
        fc = np.corrcoef(tavg_[:, :, j].T)
        err[i, j] = ((fc[r, c] - weights[r, c])**2).sum()
# look at 2nd 2s window (converges quickly)
err_ = err[-1].reshape((speeds.size, couplings.size))
# change on fc-sc metric wrt. speed & coupling strength
derr_speed = np.diff(err_.mean(axis=1)).sum()
derr_coupl = np.diff(err_.mean(axis=0)).sum()
LOG.info('derr_speed=%f, derr_coupl=%f', derr_speed, derr_coupl)
assert derr_speed > 350.0
assert derr_coupl < -500.0
