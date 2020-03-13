#     Copyright 2017 TVB-HPC contributors
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import itertools
import numpy as np
import pymbolic as pm
from dsl_cuda.tvb_hpc import model, coupling, network, utils, scheme, transforms

LOG = utils.getLogger('tvb_hpc')


def make_knl():
    # choose network model parts
    osc = model.Kuramoto()
    osc.dt = 1.0
    osc.const['omega'] = 10.0 * 2.0 * np.pi / 1e3
    cfun = coupling.Kuramoto(osc)
    cfun.param['a'] = pm.parse('a')
    scm = scheme.EulerStep(osc.dt)
    # create kernel
    knl = transforms.network_time_step(osc, cfun, scm)
    return knl, osc


def make_data():
    c = network.Connectivity.hcp0()
    return c.nnode, c.lengths, c.nnz, c.row, c.col, c.wnz, c.nz, c.weights


def run_one(args):
    j, speed, coupling, nnode, lengths, nz, nnz, row, col, wnz = args
    knl, osc = make_knl()
    lnz = (lengths[nz] / speed / osc.dt).astype(np.uintc)
    state, input, param, drift, diffs, _ = osc.prep_arrays(nnode)
    obsrv = np.zeros((lnz.max() + 3 + 4000, nnode, 2), np.float32)
    trace = np.zeros((400, nnode), np.float32)
    for i in range(trace.shape[0]):
        knl(nstep=10, nnode=nnode, ntime=obsrv.shape[0],
            state=state, input=input, param=param,
            drift=drift, diffs=diffs, obsrv=obsrv, nnz=nnz,
            delays=lnz, row=row, col=col, weights=wnz,
            a=coupling, i_step_0=i * 10)
        trace[i] = obsrv[i * 10:(i + 1) * 10, :, 0].sum(axis=0)
    return trace


def run():
    from ..numba import NumbaTarget
    utils.default_target = NumbaTarget
    nnode, lengths, nnz, row, col, wnz, nz, weights = make_data()
    # choose param space
    nc, ns = 8, 8
    couplings = np.logspace(0, 1.0, nc)
    speeds = np.logspace(0.0, 2.0, ns)
    trace = np.zeros((nc * ns, 400) + (nnode, ), np.float32)
    LOG.info('trace.nbytes %.3f MB', trace.nbytes / 2**20)
    for j, (speed, coupl) in enumerate(itertools.product(speeds, couplings)):
        run_one((j, speed, coupl, nnode, lengths, nz, nnz, row, col, wnz))

    # check correctness
    n_work_items = nc * ns
    r, c = np.triu_indices(nnode, 1)
    win_size = 200  # 2s
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


if __name__ == '__main__':
    run()
