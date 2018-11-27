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

from __future__ import division, print_function
import math as m
import numpy as _lpy_np
import numba.cuda as _lpy_ncu
import numba as _lpy_numba
from tvb_hpc import utils, network, model
from typing import List
import time


LOG = utils.getLogger('tvb_hpc')


@_lpy_ncu.jit
def Kuramoto_and_Network_and_EulerStep_inner(nstep, nnode, ntime, state, input, param, drift, diffs, obsrv, nnz, delays, row, col, weights, a, i_step_0):
    #Get the id for each thread
    tcoupling = _lpy_ncu.threadIdx.x
    tspeed = _lpy_ncu.blockIdx.x
    sid = _lpy_ncu.gridDim.x
    idp = tspeed*_lpy_ncu.blockDim.x+tcoupling
    #for each simulation step and for each node in the system
    for i_step in range(0, nstep):
        for i_node in range(0, nnode):
            #calculate the node index
            idx= idp*nnode+i_node
            #get the node params, in this case only omega
            omega = param[i_node]
            #retrieve the range of connected nodes
            j_node_lo = row[i_node]
            j_node_hi = row[i_node + 1]
            #calculate the input from other nodes at the current step
            acc_j_node = 0
            for j_node in range(j_node_lo, j_node_hi):
                acc_j_node = acc_j_node + weights[j_node]*m.sin(obsrv[((idp*ntime+((i_step + i_step_0) % ntime) + -1*delays[tspeed*nnz+j_node])*nnode+col[j_node])*2] + -1*obsrv[((idp*ntime+((i_step + i_step_0) % ntime))*nnode+i_node)*2])
            input[idx] = a[tcoupling]*acc_j_node / nnode
            #calculate the whole drift for the simulation step
            drift[idx] = omega + input[idx]
            #update the state
            state[idx] = state[idx] + drift[idx]
            #wrap the state within the desired limits 
            state[idx] = (state[idx] < 0)*(state[idx] + 6.283185307179586) + (state[idx] > 6.283185307179586)*(state[idx] + -6.283185307179586) + (state[idx] >= 0)*(state[idx] <= 6.283185307179586)*state[idx]
            theta = state[idx]
            #write the state to the observables data structure
            obsrv[((idp*ntime + ((i_step + i_step_0) % ntime))*nnode + i_node)*2 + 1] = m.sin(theta)
            obsrv[((idp*ntime + ((i_step + i_step_0) % ntime))*nnode + i_node)*2] = theta

def make_data():
    c = network.Connectivity.hcp0()
    return c.nnode, c.lengths, c.nnz, c.row, c.col, c.wnz, c.nz, c.weights

def prep_arrays(nsims, nnode: int) -> List[_lpy_np.ndarray]:
    """
    Prepare arrays for use with this model.
    """
    dtype = _lpy_np.float32
    arrs: List[_lpy_np.ndarray] = []
    for key in 'input drift diffs'.split():
        shape = nsims * nnode * 1
        arrs.append(_lpy_np.zeros(shape, dtype))
    for i, (lo, hi) in enumerate([(0, 2 * _lpy_np.pi)]):
        state = _lpy_np.ones(nsims* nnode)#.random.uniform(float(lo), float(hi),
                                        #size=(nsims* nnode ))
    arrs.append(state)
    param = _lpy_np.ones((nnode * 1), dtype)
    arrs.append(param)
    return arrs

def run_all(args):
    j, speed, coupling, nnode, lengths, nz, nnz, row, col, wnz = args
    dt = 1.0
    lnz = []
    for i in range(len(speed)): 
        lnz.append((lengths[nz] / speed[i] / dt).astype(_lpy_np.uintc))
    #print(_lpy_np.shape(lnz))
    #flat_lnz = [item for sublist in lnz for item in sublist]
    #flat_lnz = _lpy_np.asarray(flat_lnz)
    flat_lnz = _lpy_np.reshape(lnz, (nnz*len(speed)))
    input, drift, diffs, state, param = prep_arrays(len(coupling)*len(speed),nnode)
    obsrv = _lpy_np.zeros((len(coupling)*len(speed) * (max(flat_lnz) + 3 + 4000) * nnode * 2), _lpy_np.float32)
    trace = _lpy_np.zeros((len(coupling)*len(speed), 400, nnode), _lpy_np.float32) 
    threadsperblock = len(coupling)
    blockspergrid = len(speed)
    for i in range(400):
        Kuramoto_and_Network_and_EulerStep_inner[blockspergrid, threadsperblock](10, nnode, (max(flat_lnz) + 3 + 4000), state, input, param, drift, diffs, obsrv, nnz, flat_lnz, row, col, wnz, coupling, i * 10)
        o = obsrv
        o =_lpy_np.reshape(o,(len(coupling)*len(speed), (max(flat_lnz) + 3 + 4000), nnode, 2))
        trace[:,i,:] = o[:,i * 10:(i + 1) * 10, :, 0].sum(axis=1)
    return trace


def run():
    _lpy_ncu.select_device(0)
    LOG.info(_lpy_ncu.gpus)
    #utils.default_target = NumbaCudaTarget
    nnode, lengths, nnz, row, col, wnz, nz, weights = make_data()
    # choose param space
    nc, ns = 8, 8
    couplings = _lpy_np.logspace(0, 1.0, nc)
    speeds = _lpy_np.logspace(0.0, 2.0, ns)
    # Make parallel over speed anc coupling
    start = time.time()
    trace = run_all((0, speeds, couplings, nnode, lengths, nz, nnz, row, col, wnz))
    end = time.time()
    print ("Finished simulation successfully in:")
    print(end - start)
    print ("Checking correctness of results")

    # check correctness
    n_work_items = nc * ns
    r, c = _lpy_np.triu_indices(nnode, 1)
    win_size = 200  # 2s
    tavg = _lpy_np.transpose(trace, (1, 2, 0))
    win_tavg = tavg.reshape((-1, win_size) + tavg.shape[1:])
    err = _lpy_np.zeros((len(win_tavg), n_work_items))
    for i, tavg_ in enumerate(win_tavg):
        for j in range(n_work_items):
            fc = _lpy_np.corrcoef(tavg_[:, :, j].T)
            err[i, j] = ((fc[r, c] - weights[r, c])**2).sum()
    # look at 2nd 2s window (converges quickly)
    err_ = err[-1].reshape((speeds.size, couplings.size))
    # change on fc-sc metric wrt. speed & coupling strength
    derr_speed = _lpy_np.diff(err_.mean(axis=1)).sum()
    derr_coupl = _lpy_np.diff(err_.mean(axis=0)).sum()
    LOG.info('derr_speed=%f, derr_coupl=%f', derr_speed, derr_coupl)
    print (derr_speed)
    assert derr_speed > 350.0
    assert derr_coupl < -500.0
    print ("Results are correct")

if __name__ == '__main__':
    run()
