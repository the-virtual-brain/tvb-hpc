
"""
Run single simulation w/ 515k connectivity.

"""

from typing import Tuple
import numpy as np
from scipy import io, sparse
from loopy.target.c import CTarget
from tvb_hpc import model, coupling, network, utils, compiler, scheme


LOG = utils.getLogger('run515k')


def get_weights_lengths() -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
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
    return W, L



def build_components():
    LOG.info('building components')
    osc = model.G2DO()
    osc.dt = 0.1
    cfun = coupling.Linear(osc)
    net = network.Network(osc, cfun)
    scm = scheme.EulerStep(osc.dt)
    return osc, cfun, net, scm


def build_kernels(osc, net, scm):
    LOG.info('building kernels')
    target = CTarget()
    osc_knl = osc.kernel(target)
    osc_fn = compiler.CompiledKernel(osc_knl)
    net_knl = net.kernel(target)
    net_fn = compiler.CompiledKernel(net_knl)
    scm_knl = scm.kernel(target)
    scm_fn = compiler.CompiledKernel(scm_knl)
    return osc_fn, net_fn, scm_fn


if __name__ == '__main__':

    osc, cfun, net, scm = build_components()
    osc_fn, net_fn, scm_fn = build_kernels(osc, net, scm)

    W, L = get_weights_lengths()
    D = (L.data / osc.dt).astype(np.uint32)
    Dmax = D.max()
    nnode = W.shape[0]

    next, state, drift = np.zeros((3, nnode, 2), np.float32)
    input, param, diffs = np.zeros((3, nnode, 2), np.float32)
    obsrv = np.zeros((Dmax + 2, nnode, 2), np.float32)
    LOG.info('obsrv %r %.3f MB', obsrv.shape, obsrv.nbytes / 2**20)

    for i in range(10):
        with utils.timer('time step %d' % (i, )):
            net_fn(t=Dmax + 1, ntime=Dmax + 2, nnode=nnode, nnz=W.nnz,
                   row=W.indptr, col=W.indices,
                   delays=D, weights=W.data,
                   input=input, obsrv=obsrv
                   )
            osc_fn(nnode=nnode,
                   state=state, input=input, param=param,
                   drift=drift, diffs=diffs, obsrv=obsrv)
            scm_fn(nnode=nnode, nsvar=2, next=next, state=state, drift=drift)

