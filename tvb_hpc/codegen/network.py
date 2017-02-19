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

    post_template = "{post} = {cfun_post}({post}{norm});"

    def __init__(self, net: DenseNetwork):
        self.net = net
        assert isinstance(self.net, DenseNetwork)

    @property
    def kernel_name(self):
        return 'tvb_network'

    def generate_acc(self, spec, input, obsrv, i, fname):
        nvar = self.net.model.obsrv_sym.size
        nivar = self.net.model.input_sym.size
        from_idx_expr = spec.layout.generate_idx_expr('j', i, nvar)
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

    def generate_code(self, cfcg: CfunGen1, spec: BaseSpec):
        cfun = self.net.cfun
        cfun_code = cfcg.generate_code(spec)
        accs = []
        posts = []
        for i, (obsrv, pre, post, input) in enumerate(cfun.io):
            accs.append(
                self.generate_acc(
                    spec, 'input', 'obsrv', i,
                    cfcg.pre_expr_kname[str(pre)]))
            posts.append(
                self.generate_post(
                    spec, 'input',  i,
                    cfcg.post_expr_kname[str(post)]))

        inner = Loop('j', 'nnode', '\n'.join(accs))
        outer = Loop('i', 'nnode', Block(inner, '\n'.join(posts)))
        if spec.openmp:
            outer.pragma = '#pragma omp parallel for'
            inner.pragma = '#pragma omp simd'
        args = 'unsigned int nnode, '
        args += ', '.join(['{0} *{1}'.format(spec.float, name)
                           for name in 'weights input obsrv'.split()])
        self.func = Func(self.kernel_name, outer, args)
        code = self.template.format(
            cfun_code=cfun_code,
            func_code=self.func.generate_c(spec),
            **spec.dict
        )
        return code
