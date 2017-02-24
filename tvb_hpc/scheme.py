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


"""
Functions for constructing numerical schemes for nonlinear
stochastic differential equations.  Since our symbolics are
not fancy for the moment, only one-step schemes.

"""

import numpy as np
import pymbolic as pm


def euler(x, dx, dt=None):
    "Construct standard Euler step."
    dt = dt or pm.var('dt')
    return x + dt * dx


def euler_maruyama(x, f, g, dt=None, dWt=None):
    "Construct SDE Euler-Maruyama step."
    dt = dt or pm.var('dt')
    dWt = dWt or pm.var('dWt')
    return x + dt * f + dWt * dt**0.5 * g


def euler_maruyama_logp(x, f, g, xn=None, dt=None, step=euler):
    "Construct normal log p."
    dt = dt or pm.var('dt')
    mu = step(x, f, dt)
    sd = dt ** 0.5 * g
    xn = xn or np.array([pm.var(v.name + '_n') for v in x])
    return -(xn - mu) ** 2 / (2 * sd ** 2)


class EulerSchemeGen:

    template = """
{model_code}

{func_code}
"""

    model_call = """
{model_name}(nnode, state, input, param, drift, diffs, obsrv);
"""

    chunk_template = """
unsigned int idx = i * {width} * {nsvar} + j * {width} + l;
state[idx] += dt * drift[idx];
"""

    def __init__(self, model):
        self.model = model

    @property
    def kernel_name(self):
        return 'tvb_Euler_%s' % (
            self.model.__class__.__name__, )

    def generate_c(self, spec):
        model_code = self.model.generate_code(spec, Storage.static)
        model_name = self.model.kernel_name
        nsvar = self.model.state_sym.size
        chunk = Loop('l', spec.width,
                     self.chunk_template.format(nsvar=nsvar, **spec.dict))
        inner = Loop('j', nsvar, chunk)
        outer = Loop('i', 'nnode/%d' % spec.width, inner)
        if spec.openmp:
            chunk.pragma = '#pragma omp simd'
            outer.pragma = '#pragma omp parallel for'
        body = self.model_call.format(model_name=model_name)
        body += outer.generate_c(spec)
        args = 'float dt, unsigned int nnode, '
        argnames = 'state input param drift diffs obsrv'.split()
        args += ', '.join(['{float} *{0}'.format(name, **spec.dict)
                           for name in argnames])
        self.func = Func(self.kernel_name, indent(body),
                         args, storage=storage)
        return self.template.format(
            model_code=model_code,
            func_code=self.func.generate_c(spec),
            **spec.dict
        )
