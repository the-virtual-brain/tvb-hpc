import pymbolic as pm

from tvb_hpc.codegen import BaseCodeGen
from tvb_hpc.model import BaseModel


class ModelGen1(BaseCodeGen):

    template = """
#include <math.h>

void {name}(
    unsigned int nnode,
    {float} * __restrict state,
    {float} * __restrict input,
    {float} * __restrict param,
    {float} * __restrict drift,
    {float} * __restrict diffs,
    {float} * __restrict obsrv
)
{{
  {decls}
  {loop_pragma}
  for (unsigned int j=0; j < nnode; j++)
  {{
    unsigned int i = j / {width};
    unsigned int jw = j % {width};
    {body}
  }}
}}
"""

    def __init__(self, model: BaseModel):
        self.model = model

    def generate_code(self, spec):
        decls = self.generate_alignments(
            'state input param drift diffs obsrv'.split(), spec)
        decls += self.declarations(spec)
        body = self.inner_loop_lines(spec)
        code = self.template.format(
            decls='\n  '.join(decls),
            body='\n    '.join(body),
            name=self.kernel_name,
            nsvar=len(self.model.state_sym),
            # icc -> pragma vector simd ivdep, does better
            loop_pragma='#pragma omp simd safelen(%d)' % (spec['width'], ),
            **spec
        )
        if spec['float'] == 'float':
            code = code.replace('pow', 'powf')
        return code

    def declarations(self, spec):
        common = {'nsvar': len(self.model.state_sym), }
        common.update(spec)
        lines = []
        # add constants
        for name, value in self.model.const.items():
            fmt = '{float} {name} = (({float}) {value});'
            line = fmt.format(name=name, value=value, **common)
            lines.append(line)
        return lines

    def inner_loop_lines(self, spec):
        common = {
            'nsvar': len(self.model.state_sym),
        }
        common.update(spec)
        lines = []
        # unpack state, input & parameters
        fmt = '{float} {var} = {kind}[i*{nvar_width} + {isvar}*{width} + jw];'
        for kind in 'state input param'.split():
            vars = getattr(self.model, kind + '_sym')
            for i, var in enumerate(vars):
                line = fmt.format(
                    kind=kind, var=var.name, isvar=i,
                    nvar_width=len(vars)*spec['width'],
                    **common)
                lines.append(line)
        # do aux exprs
        fmt = '{float} {lhs} = {rhs};'
        for lhs, rhs in self.model.auxex:
            rhs = self.generate_c(pm.parse(rhs))
            line = fmt.format(lhs=lhs, rhs=rhs, **common)
            lines.append(line)
        # store drift / diffs / obsrv
        fmt = '{kind}[i*{nvar_width} + {isvar}*{width} + jw] = {expr};'
        for kind in 'drift diffs obsrv'.split():
            exprs = getattr(self.model, kind + '_sym')
            nvar_width = len(getattr(self.model, kind + '_sym'))*spec['width']
            for i, expr in enumerate(exprs):
                line = fmt.format(
                    kind=kind, expr=self.generate_c(expr), isvar=i,
                    nvar_width=nvar_width,
                    **common)
                lines.append(line)
        return lines

    @property
    def kernel_name(self):
        return 'tvb_' + self.model.__class__.__name__