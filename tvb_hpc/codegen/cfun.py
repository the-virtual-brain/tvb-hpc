from pymbolic.mapper.dependency import DependencyMapper

from tvb_hpc.codegen import BaseCodeGen
from tvb_hpc.coupling import BaseCoupling


class CfunGen1(BaseCodeGen):

    template = """
#include <math.h>

#pragma omp declare simd
{float} {pre_sum_name}({float} pre_syn, {float} post_syn)
{{
    {pre_decls}
    return {pre_sum};
}}

#pragma omp declare simd
{float} {post_sum_name}({float} {stat})
{{
    {post_decls}
    return {post_sum};
}}
"""

    def __init__(self, cfun: BaseCoupling):
        self.cfun = cfun

    def generate_code(self, spec):
        # TODO generalize
        assert len(self.cfun.pre_sum_sym) == 1
        assert len(self.cfun.post_sum_sym) == 1
        pre_sum = self.generate_c(self.cfun.pre_sum_sym[0])
        post_sum = self.generate_c(self.cfun.post_sum_sym[0])
        code = self.template.format(
            pre_sum_name=self.kernel_name_pre_sum,
            post_sum_name=self.kernel_name_post_sum,
            pre_sum=pre_sum,
            post_sum=post_sum,
            pre_decls=self.declarations(self.cfun.pre_sum_sym[0], spec),
            post_decls=self.declarations(self.cfun.post_sum_sym[0], spec),
            stat=self.cfun.stat,
            **spec.dict
        )
        return code

    @property
    def kernel_name_pre_sum(self):
        return 'tvb_%s_pre_sum' % (self.__class__.__name__, )

    @property
    def kernel_name_post_sum(self):
        return 'tvb_%s_post_sum' % (self.__class__.__name__, )

    def declarations(self, expr, spec):
        lines = []
        mapper = DependencyMapper(include_calls=False)
        for dep in mapper(expr):
            if dep.name in ('pre_syn', 'post_syn', 'mean', 'sum'):
                continue
            if dep.name in self.math_names:
                continue
            fmt = '{float} {name} = {value};'
            lines.append(fmt.format(
                name=dep.name,
                value=self.cfun.param[dep.name],
                **spec.dict
            ))
        return '\n    '.join(lines)