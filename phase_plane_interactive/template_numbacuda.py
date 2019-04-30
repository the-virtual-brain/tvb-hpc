# Here the pointer arithmetic only takes into account parameters which are outside of the model, in this case the coupling and the speed.
# This will be different if the parameters of the model are to be explored or if both are to be explored at the same time.


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




@_lpy_ncu.jit
def ${name}(nstep, nnode, ntime, state, input, param, nparams, drift, diffs, obsrv, nnz, delays, row, col, weights, a, i_step_0):
    idp = _lpy_ncu.blockIdx.x*_lpy_ncu.blockDim.x+_lpy_ncu.threadIdx.x
    #for each simulation step and for each node in the system
    for i_step in range(0, nstep):
        for i_node in range(0, nnode):
	    #calculate the node index
            idx= idp*nnode+i_node

        self.limit = ${limit}
        % for c in const.keys():
        self.${c} = ${c}
        % endfor

def dfun(self,state_variables):
        <% 
        sv_csl = ", ".join(sv) 
        %> 
        ${sv_csl} = state_variables

        % for c in const.keys():
        ${c} = self.${c}
        % endfor

        % for cvar in input:
        ${cvar} = 0.0
        % endfor
        
        % for i, var in enumerate(sv):
        d${var} = ${drift[i]}
        % endfor
        return [${sv_csl}] 

