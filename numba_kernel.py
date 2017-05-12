from __future__ import division, print_function

import numpy as _lpy_np
import numba as _lpy_numba

@_lpy_numba.jit
def tvb_kernel_Kuramoto_and_tvb_kernel_Network_and_tvb_kernel_EulerStep(nstep, nnode, ntime, state, input, param, drift, diffs, obsrv, nnz, delays, row, col, weights, a, i_step_0):
    for i_step in range(0, -1 + nstep + 1):
        pi = 3.141592653589793
        for i_node in range(0, -1 + nnode + 1):
            j_node_lo = row[i_node]
            state[i_node, 0] = state[i_node, 0] + -1*2*pi if state[i_node, 0] > pi else (state[i_node, 0] + 2*pi if state[i_node, 0] < -1*pi else state[i_node, 0])
            diffs[i_node, 0] = 0
            omega = param[i_node, 0]
            acc_j_node = 0
            j_node_hi = row[i_node + 1]
            theta = state[i_node, 0]
            obsrv[((i_step + i_step_0) % ntime), i_node, 1] = _lpy_np.sin(theta)
            obsrv[((i_step + i_step_0) % ntime), i_node, 0] = theta
            for j_node in range(j_node_lo, -1 + j_node_hi + 1):
                acc_j_node = acc_j_node + weights[j_node]*_lpy_np.sin(obsrv[((-1*delays[j_node] + i_step + i_step_0) % ntime), col[j_node], 0] + -1*obsrv[((i_step + i_step_0) % ntime), i_node, 0])
            input[i_node, 0] = a*acc_j_node / nnode
            I = input[i_node, 0]
            drift[i_node, 0] = omega + I
            i_svar = 0

            state[i_node, i_svar] = state[i_node, i_svar] + drift[i_node, i_svar]