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

from tvb_hpc.base import BaseKernel

class Tavg(BaseKernel):

    def __init__(self, model):
        self.model = model
        self.novar = min(len(model.input_sym), len(model.obsrv_sym))

    def kernel_domains(self):
        return "{ [i_node, i_ovar]: 0 <= i_node < nnode and 0 <= i_ovar < novar }"

    def kernel_isns(self):
        return ["tavg[i_node, i_ovar] = tavg[i_node, i_ovar] + obsrv[i_time % ntime, i_node, i_ovar]"]

    def kernel_data(self):
        return 'tavg obsrv'.split()

    def kernel_dtypes(self):
        return {'tavg,obsrv': np.float32}

tavg = Tavg(osc)
tavg_knl = tavg.kernel(target=target)
tavg_knl = lp.fix_parameters(tavg_knl, novar=tavg.novar)

# fuse kernels
knls = osc_knl, net_knl, scm_knl, tavg_knl
data_flow = [ ('input', 1, 0), ('diffs', 0, 2), ('drift', 0, 2), ('obsrv', 2, 3) ]
knl = lp.fuse_kernels(knls, data_flow=data_flow)

# and time step
knl = lp.to_batched(knl, 'nstep', [], 'i_step', sequential=True)
knl = lp.fix_parameters(knl, i_time=pm.parse('i_step + i_step_0'))
knl.args.append(lp.ValueArg('i_step_0', np.uintc))
knl = lp.add_dtypes(knl, {'i_step_0': np.uintc})

code, _ = lp.generate_code(knl)
with open('numba_kernel.py', 'w') as fd:
    fd.write(code)

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

# choose param space
ng = 8
couplings = np.logspace(1.6, 3.0, ng)
speeds = np.logspace(0.0, 2.0, ng)

trace = np.zeros((ng * ng, 400) + (nnode, ), np.float32)
LOG.info('trace.nbytes %.3f MB', trace.nbytes / 2**20)
for j, (speed, coupling) in enumerate(itertools.product(speeds, couplings)):
    lnz = (lengths[nz] / speed / osc.dt).astype(np.uintc)
    state, input, param, drift, diffs, _ = osc.prep_arrays(nnode)
    obsrv = np.zeros((lnz.max() + 3, nnode, 2), np.float32)
    for i in range(trace.shape[1]):
        knl(10, nnode, obsrv.shape[0], state, input, param,
                drift, diffs, obsrv, nnz, lnz, row, col, wnz,
                a=coupling, i_step_0=i*10)
        trace[j, i] = obsrv[i*10 % obsrv.shape[0], :, 1]
    print(j)

# check correctness
n_work_items = ng * ng
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
assert derr_speed > 500.0
assert derr_coupl < -1500.0
