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
from .model import BaseModel
from .coupling import BaseCoupling
from .utils import getLogger


LOG = getLogger(__name__)


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


class DenseDelayNetwork(DenseNetwork):
    """
    Dense network with delays.

    In TVB, we use circular indexing, but this incurs penalty in terms of
    complexity of data structures, always performing integer modulo, etc.  Far
    simpler to assume contiguous time and periodically reconstruct as memory
    constraints require, which can be done in a background thread to minimize
    overhead.

    """

    def __init__(self, model: BaseModel, cfun: BaseCoupling):
        self.model = model
        self.cfun = cfun

    def _obsrv_reshape(self, obsrv):
        # 0      1      2     3
        ntime, nblck, nvar, width = obsrv.shape
        perm = 2, 0, 1, 3
        shape = nvar, ntime, nblck * width
        obsrv_ = np.transpose(obsrv, perm).reshape(shape)
        return obsrv_

    def npeval(self, i_t, delays, weights, obsrv, input, debug=False):
        """
        Unlike DenseNetwork.npeval, obsrv here is the history of observables
        used to compute time-delayed coupling.

        Delays is expected to be array of integers; conversion from real values
        is up to the user.

        """
        nt, nb, nv, w = obsrv.shape
        nn = nb * w
        inodes = np.tile(np.r_[:nn], (nn, 1))
        obsrv_ = self._obsrv_reshape(obsrv)
        ns = {}
        for key in dir(np):
            ns[key] = getattr(np, key)
        ns.update(self.cfun.param)
        obsmat = self._npeval_mat(obsrv[i_t])
        if debug:
            self._debug(obsrv, obsrv_, nn, i_t, delays, inodes, w)
        for i, (_, pre, post, _) in enumerate(self.cfun.io):
            ns['pre_syn'] = obsrv_[i, i_t - delays, inodes]
            ns['post_syn'] = obsmat[i].reshape((-1, 1))
            weighted = eval(str(pre), ns) * weights
            ns[self.cfun.stat] = getattr(weighted, self.cfun.stat)(axis=1)
            input[:, i, :] = eval(str(post), ns).reshape((nb, w))

    def _debug(self, obsrv, obsrv_, nn, i_t, delays, inodes, w):
            LOG.debug(obsrv.shape)
            pre_syn = obsrv_[:, i_t - delays, inodes]
            for i in range(nn):
                for j in range(nn):
                    dij = (i_t - delays)[i, j]
                    for k in range(len(self.cfun.io)):
                        oij = obsrv[dij, int(j/w), k, j % w]
                        assert oij == pre_syn[k, i, j]
                        flatidx = (j % w) + obsrv.shape[-1] * (
                                k + obsrv.shape[-2] * (
                                    int(j/w) + obsrv.shape[-3]*dij))
                        LOG.debug('%d\t%d\t%d\t%d\t%d\t%f',
                                  i, j, k, dij, flatidx, oij)
