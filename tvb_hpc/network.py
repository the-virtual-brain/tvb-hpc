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
The network module provides implementations of network connectivity. Two
classes are currently provided: a dense weights matrix, without time delays,
and a dense matrix connectivity with time delays.

Sparse variants are a straightforward optimization for typical connectome
based connectivities, and they are imperative for higher resolution models
such as those based on cortical surfaces or gridded subcortical anatomy.
For the moment, they are not yet implemented.

"""

import numpy as np
import pymbolic as pm
from .base import BaseKernel
from .compiler import Spec
from .model import BaseModel
from .coupling import BaseCoupling, PostSumStat
from .utils import getLogger, subst_vars


LOG = getLogger(__name__)


class DenseNetwork(BaseKernel):
    """
    Simple dense weights network, no delays.

    """

    def __init__(self, model: BaseModel, cfun: BaseCoupling):
        self.model = model
        self.cfun = cfun

    def _insn_cfun(self, k, pre, post):
        "Generates an instruction for a single coupling function."
        # TODO add loopy hints to make more efficient
        # substitute pre_syn and post_syn for obsrv data
        pre_expr = subst_vars(
            expr=pre,
            pre_syn=pm.parse('obsrv[j, k]'),  # k -> var idx
            post_syn=pm.parse('obsrv[i, k]'),
        )
        # build weighted sum over nodes
        sum = subst_vars(
            expr=pm.parse('sum(j, weights[i, j] * pre_expr)'),
            pre_expr=pre_expr,
        )
        # mean used by some cfuns
        mean = sum / pm.var('nnode')
        # subst mean / sum through post cfun
        post_expr = subst_vars(post, sum=sum, mean=mean)
        # generate store instruction for post expr, with params
        post_expr = subst_vars(post_expr, k=k, **self.cfun.param)
        return 'input[i, %d] = %s' % ( k, post_expr )

    def kernel_isns(self):
        "Generates instructions for all cfuns"
        for k, (_, pre, post, _) in enumerate(self.cfun.io):
            yield self._insn_cfun(k, pre, post)

    def kernel_domains(self):
        return '{ [i, j]: 0 <= i, j < nnode }'

    def kernel_data(self):
        return ['nnode', 'input', 'obsrv', 'weights']

    def kernel_dtypes(self):
        return {
            'nnode': np.uintc,
            'input': np.float32,
            'obsrv': np.float32,
            'weights': np.float32,
        }



class DenseDelayNetwork(DenseNetwork):
    """
    Dense network with delays.

    In TVB, we use circular indexing, but this incurs penalty in terms of
    complexity of data structures, always performing integer modulo, etc.  Far
    simpler to assume contiguous time and periodically reconstruct as memory
    constraints require, which can be done in a background thread to minimize
    overhead. This isn't yet implemented, but to be looked at.

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

    dbg_fmt = """
printf("%d\\t%d\\t%d\\t%d\\t%d\\t%f\\n", i, j,
       {ivar}, i_t - delays[i*nnode + j],
       pre_idx_{ivar},
       obsrv[pre_idx_{ivar}]);
"""

    def build_inner(self, spec, cfun, cfcg):
        pre_idxs = []
        dbgs = []
        accs = []
        nvar = self.net.model.obsrv_sym.size
        for i, (obsrv, pre, post, input) in enumerate(cfun.io):
            pre_idxs.append('unsigned int pre_idx_{i} = {idx};'.format(
                    i=i, idx=self._pre_idx(spec, 'j', i, nvar)))
            if spec.debug:
                dbgs.append(self.dbg_fmt.format(ivar=i))
            accs.append(
                self.generate_acc(
                    spec, 'input', 'obsrv', i,
                    cfcg.pre_expr_kname[str(pre)]))
        return Loop('j', 'nnode', Block(
            '\n'.join(pre_idxs),
            '\n'.join(dbgs),
            '\n'.join(accs),
        ))

    def _pre_idx(self, spec, idx, i, nvar):
        fmt = "(i_t - delays[i*nnode + j])*(nnode * %s) + %s"
        fmt %= nvar, super()._acc_from_idx_expr(spec, idx, i, nvar)
        return fmt

    def _acc_from_idx_expr(self, spec, idx, i, nvar):
        return 'pre_idx_{i}'.format(i=i)

    def _base_args(self):
        extra = ['unsigned int i_t', 'unsigned int *delays']
        return super()._base_args() + extra
