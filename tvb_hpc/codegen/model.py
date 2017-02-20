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

import pymbolic as pm

from .base import BaseCodeGen, BaseSpec, Storage, Loop, indent, Func
from ..model import BaseModel


class ModelGen1(BaseCodeGen):

    template = "#include <math.h>\n{func_code}"

    def __init__(self, model: BaseModel):
        self.model = model
        self.storage = Storage.default

    def generate_code(self, spec: BaseSpec, storage=Storage.default):
        decls = self.generate_alignments(
            'state input param drift diffs obsrv'.split(), spec)
        decls += self.declarations(spec)
        if self.model.param_sym.size == 0:
            decls.append('(void) param;')
        loop_body = self.inner_loop_lines(spec)
        inner = Loop('j', spec.width, '\n'.join(loop_body))
        outer = Loop('i', 'nnode / %d' % (spec.width, ), inner)
        if spec.openmp:
            # icc -> pragma vector simd ivdep, does better?
            inner.pragma = '#pragma omp simd'
            outer.pragma = '#pragma omp parallel for'
        body = '{decls}\n{loops}'.format(
            decls='\n  '.join(decls),
            loops=indent(outer.generate_c(spec)),
        )
        argnames = 'state input param drift diffs obsrv'
        args = 'unsigned int nnode, '
        args += ', '.join(['{float} *{name}'.format(name=name, **spec.dict)
                           for name in argnames.split()])
        self.func = Func(self.kernel_name, body, args, storage=storage)
        code = self.template.format(
            func_code=self.func.generate_c(spec)
        )
        return code

    def declarations(self, spec):
        common = {'nsvar': len(self.model.state_sym), }
        common.update(spec.dict)
        lines = []
        # add constants
        for name, value in self.model.const.items():
            if name in self.model.param:
                continue
            fmt = '{float} {name} = (({float}) {value});'
            line = fmt.format(name=name, value=value, **common)
            lines.append(line)
        return lines

    def inner_loop_lines(self, spec: BaseSpec):
        common = {
            'nsvar': len(self.model.state_sym),
        }
        common.update(spec.dict)
        lines = []
        # unpack state, input & parameters
        fmt = '{float} {var} = {kind}[{idx_expr}];'
        for kind in 'state input param'.split():
            vars = getattr(self.model, kind + '_sym')
            for i, var in enumerate(vars):
                idx_expr = 'i * {nvar} * {width} + {i} * {width} + j'
                line = fmt.format(
                    kind=kind, var=var.name,
                    idx_expr=idx_expr.format(
                        nvar=len(vars),
                        width=spec.width,
                        i=i
                    ),
                    **common)
                lines.append(line)
        # do aux exprs
        fmt = '{float} {lhs} = {rhs};'
        for lhs, rhs in self.model.auxex:
            rhs = self.generate_c(pm.parse(rhs), spec)
            line = fmt.format(lhs=lhs, rhs=rhs, **common)
            lines.append(line)
        # store drift / diffs / obsrv
        fmt = '{kind}[{idx_expr}] = {expr};'
        for kind in 'drift diffs obsrv'.split():
            exprs = getattr(self.model, kind + '_sym')
            nvar = len(getattr(self.model, kind + '_sym'))
            for i, expr in enumerate(exprs):
                idx_expr = 'i * {nvar} * {width} + {i} * {width} + j'
                line = fmt.format(
                    kind=kind,
                    expr=self.generate_c(expr, spec),
                    idx_expr=idx_expr.format(
                        nvar=nvar,
                        width=spec.width,
                        i=i
                    ),
                    **common)
                lines.append(line)
        return lines

    @property
    def kernel_name(self):
        return 'tvb_' + self.model.__class__.__name__
