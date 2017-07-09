import numpy as np
import loopy as lp
from tvb_hpc.numbacudatarget import NumbaCudaTarget
target = NumbaCudaTarget()

coupling_raw = """
<> theta_i = state[i_nodie] {nosync=*}	
<> sum = 0.0	
<> wij = weights[j_node]
if wij != 0.0
    <int> dij = lengths[j_node] * rec_speed_dt		
    <float32> theta_j = state[j_node] {nosync=*}
    sum = sum + wij * sin(theta_j - theta_i)
end
"""
model_raw = """
theta_i = theta_i + dt * (omega + coupling_value * rec_n * sum) 	
theta_i = theta_i + (sig * rand) 					
theta_i = wrap_2_pi(theta_i) 					
tavg[i_node] = tavg[i_node] + sin(theta_i) 	
state[i_node] = theta_i {nosync=*}	
"""
node = 10
#Raw coupling
couplingknl = lp.make_kernel("{ [i_node, j_node]: 0<=i_node, j_node<n_node}",coupling_raw,target = target)
couplingknl = lp.add_dtypes(couplingknl, {"lengths": np.float32, "state": np.float32, "weights": np.float32, "theta_i": np.float32, "rec_speed_dt": np.float32})
couplingknl = lp.split_iname(couplingknl, "j_node", 1, outer_tag='l.0')
#Raw model
modelknl = lp.make_kernel("{ [i_node]: 0<=i_node<n_node}",model_raw,target = target)
modelknl = lp.add_dtypes(modelknl, {"state": np.float32, "theta_i": np.float32, "tavg": np.float32})
# Fuse
knls = couplingknl, modelknl
data_flow = [('state', 0, 1)]
knl = lp.fuse_kernels(knls, data_flow = data_flow)
print (knl)
print ("****")
knl = lp.split_iname(knl, "i_node", 128, outer_tag='g.0')
print(lp.generate_code_v2(knl).all_code())

