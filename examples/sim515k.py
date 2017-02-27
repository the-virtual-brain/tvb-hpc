
"""
Run single simulation w/ 515k connectivity.

"""

from typing import Tuple
import loopy as lp
lp.set_caching_enabled(False)
import numpy as np
from scipy import io, sparse
from loopy.target.c import CTarget, CASTBuilder
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



class MyCASTBuilder(CASTBuilder):
    def emit_sequential_loop(self, codegen_state, iname, iname_dtype,
            lbound, ubound, inner):
        from cgen import Pragma, Block
        loop = super().emit_sequential_loop(codegen_state, iname, iname_dtype,
            lbound, ubound, inner)
        if iname == 'i':
            return Block(contents=[
                Pragma('omp parallel for'),
                loop,
            ])
        return loop


class MyCTarget(CTarget):
    def get_device_ast_builder(self):
        return MyCASTBuilder(self)

def build_kernels(osc, net, scm):
    LOG.info('building kernels')
    target = MyCTarget()

    # TODO make compiler reusable..
    def comp():
        comp = compiler.Compiler(cc='/usr/local/bin/gcc-6')
        comp.cflags += '-fopenmp -march=native -ffast-math'.split()
        comp.ldflags += '-fopenmp'.split()
        return comp

    osc_knl = osc.kernel(target)
    osc_fn = compiler.CompiledKernel(osc_knl, comp())

    net_knl = net.kernel(target)
    net_fn = compiler.CompiledKernel(net_knl, comp())

    scm_knl = scm.kernel(target)
    scm_knl = lp.prioritize_loops(scm_knl, ['i', 'j'])
    scm_knl = lp.fix_parameters(scm_knl, nsvar=2)
    scm_knl = lp.tag_inames(scm_knl, [('j', 'ilp')])
    scm_fn = compiler.CompiledKernel(scm_knl, comp())
    return osc_fn, net_fn, scm_fn, comp


if __name__ == '__main__':

    osc, cfun, net, scm = build_components()
    osc_fn, net_fn, scm_fn, comp = build_kernels(osc, net, scm)

    W, L = get_weights_lengths()
    D = (L.data / 1.0).astype(np.uint32)
    Dmax = D.max()
    nnode = W.shape[0]

    next, state, drift = np.zeros((3, nnode, 2), np.float32)
    input, param, diffs = np.zeros((3, nnode, 2), np.float32)
    obsrv = np.zeros((Dmax + 2, nnode, 2), np.float32)
    LOG.info('obsrv %r %.3f MB', obsrv.shape, obsrv.nbytes / 2**20)

    with utils.timer('10x sp mat vec'):
        vec = obsrv[0, :, 0].copy()
        for i in range(10):
            W * vec

    for i in range(10):
        with utils.timer('time step %d' % (i, )):
            # net kernel dominates run time
            net_fn(t=Dmax + 1, ntime=Dmax + 2, nnode=nnode, nnz=W.nnz,
                   row=W.indptr, col=W.indices,
                   delays=D, weights=W.data,
                   input=input, obsrv=obsrv
                   )
            osc_fn(nnode=nnode,
                   state=state, input=input, param=param,
                   drift=drift, diffs=diffs, obsrv=obsrv)
            scm_fn(nnode=nnode, nsvar=2, next=next, state=state, drift=drift)

