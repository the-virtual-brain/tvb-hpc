#     Copyright 2018 TVB-HPC contributors
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

import numpy as np
import loopy as lp
import pymbolic as pm
from dsl_cuda.tvb_hpc import model, coupling, network, utils, scheme

LOG = utils.getLogger('tvb_hpc')


def network_time_step(
        model: model.BaseKernel,
        coupling: coupling.BaseCoupling,
        scheme: scheme.TimeStepScheme,
        target: lp.target.TargetBase=None,
        ):
    target = target or utils.default_target()
    # fuse kernels
    kernels = [
        model.kernel(target),
        network.Network(model, coupling).kernel(target),
        lp.fix_parameters(scheme.kernel(target), nsvar=len(model.state_sym)),
    ]
    data_flow = [
        ('input', 1, 0),
        ('diffs', 0, 2),
        ('drift', 0, 2),
        ('state', 2, 0)
    ]
    knl = lp.fuse_kernels(kernels, data_flow=data_flow)
    # time step
    knl = lp.to_batched(knl, 'nstep', [], 'i_step', sequential=True)
    new_i_time = pm.parse('(i_step + i_step_0) % ntime')
    knl = lp.fix_parameters(knl, i_time=new_i_time)
    knl.args.append(lp.ValueArg('i_step_0', np.uintc))
    knl = lp.add_dtypes(knl, {'i_step_0': np.uintc})
    return knl
