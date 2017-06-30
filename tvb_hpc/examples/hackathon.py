# WIP toward hackathon kernel

import itertools
import numpy as np
import loopy as lp
import pymbolic as pm
from scipy import sparse
from tvb_hpc import model, coupling, network, utils, scheme
from tvb_hpc.numba import NumbaTarget

LOG = utils.getLogger('tvb_hpc')


def make_knl():

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
    data_flow = [('input', 1, 0),
                 ('diffs', 0, 2),
                 ('drift', 0, 2),
                 ('state', 2, 0)]
    knl = lp.fuse_kernels(knls, data_flow=data_flow)

    # and time step
    knl = lp.to_batched(knl, 'nstep', [], 'i_step', sequential=True)
    knl = lp.fix_parameters(knl, i_time=pm.parse('(i_step + i_step_0) % ntime'))
    knl.args.append(lp.ValueArg('i_step_0', np.uintc))
    knl = lp.add_dtypes(knl, {'i_step_0': np.uintc})

    return knl, osc

def make_data():
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
    return nnode, lengths, nnz, row, col, wnz, nz, weights


def run_one(args):
    j, speed, coupling, nnode, lengths, nz, nnz, row, col, wnz = args
    knl, osc = make_knl()
    lnz = (lengths[nz] / speed / osc.dt).astype(np.uintc)
    state, input, param, drift, diffs, _ = osc.prep_arrays(nnode)
    obsrv = np.zeros((lnz.max() + 3 + 4000, nnode, 2), np.float32)
    trace = np.zeros((400, nnode), np.float32)
    for i in range(trace.shape[0]):
        knl(10, nnode, obsrv.shape[0], state, input, param,
            drift, diffs, obsrv, nnz, lnz, row, col, wnz,
            a=coupling, i_step_0=i * 10)
        trace[i] = obsrv[i * 10:(i + 1) * 10, :, 0].sum(axis=0)
    return trace

def run():
    nnode, lengths, nnz, row, col, wnz, nz, weights = make_data()
    # choose param space
    nc, ns = 8, 8
    couplings = np.logspace(0, 1.0, nc)
    speeds = np.logspace(0.0, 2.0, ns)
    trace = np.zeros((nc * ns, 400) + (nnode, ), np.float32)
    LOG.info('trace.nbytes %.3f MB', trace.nbytes / 2**20)
    args = []
    for j, (speed, coupling) in enumerate(itertools.product(speeds, couplings)):
        args.append(
            (j, speed, coupling, nnode, lengths, nz, nnz, row, col, wnz)
        )
    if False:
        from multiprocessing import Pool
        with Pool() as pool:
            trace = np.array(pool.map(run_one, args))
    else:
        trace = np.array([run_one(_) for _ in args])

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
    plot_err(err)
    # look at 2nd 2s window (converges quickly)
    err_ = err[-1].reshape((speeds.size, couplings.size))
    # change on fc-sc metric wrt. speed & coupling strength
    derr_speed = np.diff(err_.mean(axis=1)).sum()
    derr_coupl = np.diff(err_.mean(axis=0)).sum()
    LOG.info('derr_speed=%f, derr_coupl=%f', derr_speed, derr_coupl)
    assert derr_speed > 350.0
    assert derr_coupl < -500.0


def plot_err(err):
    import matplotlib as mpl
    mpl.use('Agg')
    import pylab as pl
    pl.imshow(err)
    pl.savefig('hackathon.png')


if __name__ == '__main__':
    run()
