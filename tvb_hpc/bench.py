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
A few benchmarks, not comprehensive.

"""

import numpy as np
from .lab import HMJE, LinearCfun, DenseNetwork, timer, SimpleTimeStep
from .compiler import Spec
from .utils import getLogger


class BaseBench:
    def __init__(self):
        self.logger = getLogger(self.__class__.__name__)


class CtypesOverhead(BaseBench):
    """
    This benchmark tests the amount of overhead incurred by using ctypes.

    """

    def et(self, nnode, niter):
        model = HMJE()
        cfun = LinearCfun(model)
        weights = np.abs(np.random.randn(nnode, nnode)).astype('f')
        net = DenseNetwork(model, cfun)
        spec = Spec('float', 2)
        stepper = SimpleTimeStep(model, cfun, net, spec=spec)
        stepper.prep_data(weights=weights)
        with timer() as t:
            stepper.run(n_iter=niter)
        return t.elapsed

    def run(self):
        # consider 2 node call as overhead
        niter = 1000
        et2 = self.et(2, niter) / niter
        fmt = 'nnode=%d, overhead per call %f %%.'
        for nnode in 2**np.r_[1:10]:
            et_per = self.et(nnode, niter) / niter
            self.logger.info(fmt, nnode, 100*et2/et_per)


def main(argv):
    _, name = argv
    Bench = globals().get(name)
    bench = Bench()
    bench.run()


if __name__ == '__main__':
    import sys
    main(sys.argv)
