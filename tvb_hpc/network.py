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

import numpy as np
from tvb_hpc.model import BaseModel
from tvb_hpc.coupling import BaseCoupling


class DenseNetwork:
    """
    Simple dense weights network, no delays etc.

    """

    def __init__(self, model: BaseModel, cfun: BaseCoupling):
        self.model = model
        self.cfun = cfun

    def _npeval_mat(self, a):
        if a.shape[1] == 0:
            return []
        return np.transpose(a, (1, 0, 2)).reshape((a.shape[1], -1))

    def npeval(self, weights, obsrv, input):
        """
        Evaluate network (obsrv -> weights*cfun -> input) on arrays
        using NumPy.

        """
        # TODO generalize layout.. xarray?
        nn, _, w = obsrv.shape
        ns = {}
        for key in dir(np):
            ns[key] = getattr(np, key)
        ns.update(self.cfun.param)
        obsmat = self._npeval_mat(obsrv)
        for i, (_, pre, post, _) in enumerate(self.cfun.io):
            ns['pre_syn'] = obsmat[i]
            ns['post_syn'] = obsmat[i].reshape((-1, 1))
            weighted = eval(str(pre), ns) * weights
            ns[self.cfun.stat] = getattr(weighted, self.cfun.stat)(axis=1)
            input[:, i, :] = eval(str(post), ns).reshape((nn, w))
