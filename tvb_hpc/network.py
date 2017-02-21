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
from .codegen import BaseCodeGen, Loop, Func, Block, Storage
from .compiler import Spec
from .model import BaseModel
from .coupling import BaseCoupling
from .utils import getLogger


LOG = getLogger(__name__)


class DenseNetwork(BaseCodeGen):
    """
    Simple dense weights network, no delays.

    """

    template = "{cfun_code}\n{func_code}"

    acc_template = "{acc} += {wij}*{cfun_pre}({pre_syn}, {post_syn});"

    zero_template = "{acc} = (({float}) 0);"

    post_template = "{post} = {cfun_post}({post}{norm});"

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

    @property
    def kernel_name(self):
        return 'tvb_network'

    def _acc_from_idx_expr(self, spec, idx, i, nvar):
        return spec.layout.generate_idx_expr('j', i, nvar)

    def generate_zero(self, spec, input, i):
        nivar = self.model.input_sym.size
        acc_idx_expr = spec.layout.generate_idx_expr('i', i, nivar)
        return self.zero_template.format(
            acc='%s[%s]' % (input, acc_idx_expr),
            float=spec.float)

    def generate_acc(self, spec, input, obsrv, i, fname):
        nvar = self.model.obsrv_sym.size
        nivar = self.model.input_sym.size
        from_idx_expr = self._acc_from_idx_expr(spec, 'j', i, nvar)
        to_idx_expr = spec.layout.generate_idx_expr('i', i, nvar)
        acc_idx_expr = spec.layout.generate_idx_expr('i', i, nivar)
        return self.acc_template.format(
            wij='weights[i*nnode + j]',
            acc='%s[%s]' % (input, acc_idx_expr),
            pre_syn='%s[%s]' % (obsrv, from_idx_expr),
            post_syn='%s[%s]' % (obsrv, to_idx_expr),
            cfun_pre=fname)

    def generate_post(self, spec, input, i, fname):
        nivar = self.model.input_sym.size
        out_idx_expr = spec.layout.generate_idx_expr('i', i, nivar)
        norm = ''
        if self.cfun.stat == 'mean':
            norm = ' / nnode'
        return self.post_template.format(
            post='%s[%s]' % (input, out_idx_expr),
            norm=norm,
            cfun_post=fname)

    def generate_c(self, *args):
        return self.generate_code(*args)

    def build_inner(self, spec, cfun):
        accs = []
        for i, (obsrv, pre, post, input) in enumerate(cfun.io):
            accs.append(
                self.generate_acc(
                    spec, 'input', 'obsrv', i,
                    cfun.pre_expr_kname[str(pre)]))
        return Loop('j', 'nnode', '\n'.join(accs))

    def generate_code(self, spec: Spec,
                      storage: Storage=Storage.default):
        cfun = self.cfun
        cfun_code = cfun.generate_code(spec)
        posts = []
        zeros = []
        for i, (obsrv, pre, post, input) in enumerate(cfun.io):
            zeros.append(self.generate_zero(spec, 'input', i))
            posts.append(
                self.generate_post(
                    spec, 'input',  i,
                    cfun.post_expr_kname[str(post)]))
        inner = self.build_inner(spec, cfun)
        outer = Loop('i', 'nnode',
                     Block('\n'.join(zeros),
                           inner,
                           '\n'.join(posts)))
        if spec.openmp:
            outer.pragma = '#pragma omp parallel for'
            inner.pragma = '#pragma omp simd'
        args = self._base_args()
        args += ['{0} *{1}'.format(spec.float, name)
                 for name in 'weights input obsrv'.split()]
        args = ', '.join(args)
        self.func = Func(self.kernel_name, outer, args,
                         storage=storage)
        code = self.template.format(
            cfun_code=cfun_code,
            func_code=self.func.generate_c(spec),
            **spec.dict
        )
        return code

    def _base_args(self):
        return ['unsigned int nnode']


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
