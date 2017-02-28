
"""
Parallel simulation of 105 connectomes from a multiple schlerosis study.

A lot of this is data munging for the moment.

"""

import os.path
import numpy as np
import loopy as lp
from scipy import sparse
lp.set_caching_enabled(False)
from tvb_hpc import model, coupling, network, utils, compiler, scheme

LOG = utils.getLogger('sep645')

import glob

# load data, cache as npz
sep_npz_fname = 'data/sep.npz'
if not os.path.exists(sep_npz_fname):
    LOG.info('reading from txt files..')
    W = np.array(utils.loadtxt_many('data/sep/*/*_N.txt'))
    L = np.array(utils.loadtxt_many('data/sep/*/*_dist.txt'))
    np.savez(sep_npz_fname, W=W, L=L)
else:
    LOG.info('reading from npz')
    npz = np.load(sep_npz_fname)
    W = npz['W']
    L = npz['L']
assert W.shape == L.shape
assert W.shape[1:] == (645, 645)

scaling = utils.scaling(W)
LOG.info('scaling of weights seems %s', scaling)
if scaling == 'log':
    LOG.info('log transforming weights')
    W = np.log(W + 1)

# drop subcortical; too variable
W = W[:, 36:, 36:].astype(np.float32)
L = L[:, 36:, 36:].astype(np.float32)

# drop below, clip above (0.5 to improve sparsity / reduce noise)
q99 = np.percentile(W.flat[:], 99)
qlo = 0.5 * q99
LOG.info('W dropping below %f, clipping at %f', qlo, q99)
W[W<qlo] = 0
W[W>q99] = q99

nsubj, nnode, _ = W.shape
LOG.info('nsubj %d nnode %d', nsubj, nnode)

# impose identical group sparsity
W_nz: np.ndarray = ~(W == 0)
W_nzmask = W_nz.any(axis=0)

W_ = np.array([((w + 1e-3) * W_nzmask) for w in W])
L_ = np.array([((l + 1e-3) * W_nzmask) for l in L])

W0_nz: np.ndarray = ~(W_[0] == 0)
W_nnz = W0_nz.sum()
LOG.info('W nnz %.3f', W_nnz * 100 / (nnode * nnode))

# build group sparse W & L
sW_data = np.zeros((nsubj, W_nnz), np.float32)
sL_data = np.zeros((nsubj, W_nnz), np.float32)
sW_col = np.zeros((W_nnz, ), np.uintc)
sW_row = np.zeros((nnode + 1, ), np.uintc)
for i, (w, l) in enumerate(zip(W_, L_)):
    sw = sparse.csr_matrix(w)
    sl = sparse.csr_matrix(l)
    np.testing.assert_allclose(sw.nnz, W_nnz)
    np.testing.assert_allclose(sl.nnz, W_nnz)
    sL_data[i, :] = sl.data
    sW_data[i, :] = sl.data
    if i == 0:
        sW_col[:] = sw.indices
        sW_row[:] = sw.indptr
    else:
        np.testing.assert_allclose(sw.indices, sW_col)
        np.testing.assert_allclose(sw.indptr, sW_row)
    np.testing.assert_allclose(sl.indices, sW_col)
    np.testing.assert_allclose(sl.indptr, sW_row)
sD_data = (sL_data / 0.1).astype('i')
Dmax = sD_data.max()



# build kernels
target = compiler.OpenMPCTarget()
target.iname_pragma_map['i_subj'] = 'omp parallel for'

def comp():
    comp = compiler.Compiler(cc='/usr/local/bin/gcc-6')
    comp.cflags += '-fopenmp -march=native -ffast-math'.split()
    comp.ldflags += '-fopenmp'.split()
    return comp


def batch_knl(knl):
    varying = 'weights delays state input obsrv drift diffs next'.split()
    # wait for bug fix
    #varying.remove('delays')
    return lp.to_batched(knl, 'nsubj', varying, 'i_subj',
                         sequential=True)

osc = model.HMJE()
osc.dt = 0.1
osc_knl = osc.kernel(target)
osc_knl = batch_knl(osc_knl)
osc_fn = compiler.CompiledKernel(osc_knl, comp())

cfun = coupling.Diff(osc)
net = network.Network(osc, cfun)
net_knl = net.kernel(target)
net_knl = batch_knl(net_knl)
net_fn = compiler.CompiledKernel(net_knl, comp())

scm = scheme.EulerStep(osc.dt)
scm_knl = scm.kernel(target)
scm_knl = batch_knl(scm_knl)
scm_knl = lp.prioritize_loops(scm_knl, ['i_subj', 'i', 'j'])
scm_knl = lp.fix_parameters(scm_knl, nsvar=len(osc.state_sym))
scm_knl = lp.tag_inames(scm_knl, [('j', 'ilp')])
scm_fn = compiler.CompiledKernel(scm_knl, comp())

# build other data arrays
next, state, drift = np.zeros((3, nsubj, nnode, osc.state_sym.size), np.float32)
input, param, diffs = np.zeros((3, nsubj, nnode, 6), np.float32)
obsrv = np.zeros((Dmax + 3, nsubj, nnode, 2), np.float32)
LOG.info('obsrv %r %.3f MB', obsrv.shape, obsrv.nbytes / 2**20)


# step function
def step(n_step=1):
    for _ in range(n_step):
        t = Dmax + 1
        net_fn(nsubj=nsubj, t=t, ntime=Dmax + 3, nnode=nnode, nnz=W_nnz,
               row=sW_row, col=sW_col,
               delays=sD_data, weights=sW_data,
               input=input, obsrv=obsrv
               )
        osc_fn(nsubj=nsubj, nnode=nnode,
               state=state, input=input, param=param,
               drift=drift, diffs=diffs, obsrv=obsrv[t])
        scm_fn(nsubj=nsubj, nnode=nnode, nsvar=2, next=next, state=state,
               drift=drift)
    # TODO
    # ballon_fn()
    # cov_fn()
    # obsrv[:Dmax] = obsrv[-Dmax:]

# warm up
step(20)

# time it (45 s for G2DO, 50 RWW, 100 Kura)
# 1000 -> 100 ms, need 5 min BOLD, 300s, 3000 * 45s / 3600 -> 37h..
with utils.timer('1000 time steps'):
    step(1000)


# TODO check against _fmri