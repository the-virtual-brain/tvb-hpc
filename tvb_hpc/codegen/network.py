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

from .base import BaseCodeGen, BaseSpec, Loop, Func, Block
from .cfun import CfunGen1
from ..network import DenseNetwork


class NetGen1(BaseCodeGen):
    """
    Provides basic code generation for networks.

    """

    template = "{cfun_code}\n{func_code}"

    acc_template = "{acc} += {wij}*{cfun_pre}({pre_syn}, {post_syn});"

    zero_template = "{acc} = (({float}) 0);"

    post_template = "{post} = {cfun_post}({post}{norm});"

    def __init__(self, net: DenseNetwork):
        self.net = net

    @property
    def kernel_name(self):
        return 'tvb_network'

    def _acc_from_idx_expr(self, spec, idx, i, nvar):
        return spec.layout.generate_idx_expr('j', i, nvar)

    def generate_zero(self, spec, input, i):
        nivar = self.net.model.input_sym.size
        acc_idx_expr = spec.layout.generate_idx_expr('i', i, nivar)
        return self.zero_template.format(
            acc='%s[%s]' % (input, acc_idx_expr),
            float=spec.float)

    def generate_acc(self, spec, input, obsrv, i, fname):
        nvar = self.net.model.obsrv_sym.size
        nivar = self.net.model.input_sym.size
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
        nivar = self.net.model.input_sym.size
        out_idx_expr = spec.layout.generate_idx_expr('i', i, nivar)
        norm = ''
        if self.net.cfun.stat == 'mean':
            norm = ' / nnode'
        return self.post_template.format(
            post='%s[%s]' % (input, out_idx_expr),
            norm=norm,
            cfun_post=fname)

    def generate_c(self, *args):
        return self.generate_code(*args)

    def build_inner(self, spec, cfun, cfcg):
        accs = []
        for i, (obsrv, pre, post, input) in enumerate(cfun.io):
            accs.append(
                self.generate_acc(
                    spec, 'input', 'obsrv', i,
                    cfcg.pre_expr_kname[str(pre)]))
        return Loop('j', 'nnode', '\n'.join(accs))

    def generate_code(self, cfcg: CfunGen1, spec: BaseSpec):
        cfun = self.net.cfun
        cfun_code = cfcg.generate_code(spec)
        posts = []
        zeros = []
        for i, (obsrv, pre, post, input) in enumerate(cfun.io):
            zeros.append(self.generate_zero(spec, 'input', i))
            posts.append(
                self.generate_post(
                    spec, 'input',  i,
                    cfcg.post_expr_kname[str(post)]))
        inner = self.build_inner(spec, cfun, cfcg)
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
        self.func = Func(self.kernel_name, outer, args)
        code = self.template.format(
            cfun_code=cfun_code,
            func_code=self.func.generate_c(spec),
            **spec.dict
        )
        return code

    def _base_args(self):
        return ['unsigned int nnode']


class NetGen2(NetGen1):
    """
    Code generation handling time delay coupling.

    """

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
