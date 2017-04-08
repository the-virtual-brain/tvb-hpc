# WIP toward hackathon kernel

import numpy as np
import loopy as lp
from scipy import sparse
lp.set_caching_enabled(False)
from tvb_hpc import model, coupling, network, utils, compiler, scheme
from tvb_hpc.numba import NumbaTarget

LOG = utils.getLogger('hackathon')

target = NumbaTarget()

osc = model.Kuramoto()
osc.dt = 0.1
osc_knl = osc.kernel(target)

cfun = coupling.Kuramoto(osc)
net = network.Network(osc, cfun)
net_knl = net.kernel(target)

scm = scheme.EulerStep(osc.dt)
scm_knl = scm.kernel(target)
scm_knl = lp.fix_parameters(scm_knl, nsvar=len(osc.state_sym))

knls = osc_knl, net_knl, scm_knl
data_flow = [
    ('input', 1, 0),
    ('diffs', 0, 2),
    ('drift', 0, 2),
]
knl = lp.fuse_kernels(knls, data_flow=data_flow)

knl = lp.to_batched(knl, 'nstep', [], 'i_step', sequential=True)

# TODO next -> state, i_step -> i_time

target.get_kernel_executor(knl)
