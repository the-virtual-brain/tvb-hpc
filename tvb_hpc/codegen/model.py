import pymbolic as pm

from .base import BaseCodeGen, BaseSpec, Storage
from ..model import BaseModel


class ModelGen1(BaseCodeGen):

    template = """
#include <math.h>

{storage}
void {name}(
    unsigned int nnode,
    {float} *state,
    {float} *input,
    {float} *param,
    {float} *drift,
    {float} *diffs,
    {float} *obsrv
)
{{
  {decls}
  {loop_pragma}
  for (unsigned int j=0; j < nnode; j++)
  {{
    {body}
  }}
}}
"""

    def __init__(self, model: BaseModel):
        self.model = model
        self.storage = Storage.default

    def generate_code(self, spec: BaseSpec, storage=None):
        decls = self.generate_alignments(
            'state input param drift diffs obsrv'.split(), spec)
        decls += self.declarations(spec)
        body = self.inner_loop_lines(spec)
        loop_pragma = ''
        if spec.openmp:
            loop_pragma = '#pragma omp simd'
        code = self.template.format(
            decls='\n  '.join(decls),
            body='\n    '.join(body),
            name=self.kernel_name,
            nsvar=len(self.model.state_sym),
            # icc -> pragma vector simd ivdep, does better
            loop_pragma=loop_pragma,
            storage=(storage or self.storage).value,
            **spec.dict
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
                idx_expr = spec.layout.generate_idx_expr('j', i, len(vars))
                line = fmt.format(
                    kind=kind, var=var.name, idx_expr=idx_expr,
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
                idx_expr = spec.layout.generate_idx_expr('j', i, nvar)
                line = fmt.format(
                    kind=kind,
                    expr=self.generate_c(expr, spec),
                    idx_expr=idx_expr,
                    **common)
                lines.append(line)
        return lines

    @property
    def kernel_name(self):
        return 'tvb_' + self.model.__class__.__name__
