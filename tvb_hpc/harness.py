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
Harnesses coordinate multiple kernels for a given use case.

While time stepping is the only harness currently implemented,
two important use cases to be added are parallel parameter sweeps
and log-p & gradients for Bayesian inversion.

"""

from .compiler import Compiler, Spec
from .utils import getLogger
from .scheme import EulerSchemeGen


class BaseHarness:
    "Type tag."


class SimpleTimeStep(BaseHarness):
    """
    Simple time-stepping for a single simulation.

    """

    def __init__(self, model, cfun, net, spec=None, comp=None):
        self.model = model
        self.cfun = cfun
        self.net = net
        self.step = EulerSchemeGen(model)
        # TODO improve API on this stuff
        spec = spec or Spec(openmp=True)
        comp = comp or Compiler(openmp=True)
        comp.cflags = comp.cflags + '-O3 -ffast-math -march=native'.split()
        net_c = self.net.generate_c(spec)
        self.net.func.fn = getattr(comp('net', net_c),
                                   self.net.kernel_name)
        self.net.func.annot_types(self.net.func.fn)
        step_c = self.step.generate_c(spec)
        self.step.func.fn = getattr(comp('step', step_c),
                                    self.step.kernel_name)
        self.step.func.annot_types(self.step.func.fn)
        self.spec = spec
        self.comp = comp
        self.logger = getLogger(self.__class__.__name__)

    def prep_data(self, weights):
        self.weights = weights.astype(self.spec.np_dtype)
        self.nnode = weights.shape[0]
        self.nblock = int(self.nnode / self.spec.width)
        self.arrs = self.model.prep_arrays(self.nblock, self.spec)
        self.logger.debug('nblock %d', self.nblock)
        self.logger.debug('array shape %r', self.arrs[0].shape)

    @property
    def obsrv(self):
        return self.arrs[-1].copy()

    def run(self, n_iter=100):
        dt = self.model.dt
        x, i, p, f, g, o = self.arrs
        for _ in range(n_iter):
            self.net.func(self.nnode, self.weights, i, o)
            self.step.func(dt, self.nnode, x, i, p, f, g, o)
