from pymbolic.mapper.dependency import DependencyMapper

from .base import BaseCodeGen, BaseSpec, Storage
from . import literal
from tvb_hpc.coupling import BaseCoupling


class CfunGen1(BaseCodeGen):

    template = """
#include <math.h>
{pre_funcs}
{post_funcs}
"""

    pre_template = """
{func_pragma}
{storage} {float} {pre_sum_name}({float} pre_syn, {float} post_syn)
{{
    {pre_decls}
    return {pre_sum};
}}
"""

    post_template = """
{func_pragma}
{storage} {float} {post_sum_name}({float} {stat})
{{
    {post_decls}
    return {post_sum};
}}
"""

    def __init__(self, cfun: BaseCoupling,
                 storage: Storage=Storage.static):
        self.cfun = cfun
        self.kernel_names = []
        self.storage = storage

    def func_pragma(self, spec):  # {{{
        func_pragma = ''
        if spec.openmp:
            func_pragma = '#pragma omp declare simd'
        return func_pragma  # }}}

    # TODO use classes for pre/psot functions, so we can have API
    # to ask things like, does it use post_syn value etc..

    def generate_pre(self, i, expr, spec: BaseSpec):  # {{{
        expr_c = self.generate_c(expr, spec)
        pre_decls = ''
        if 'post_syn' not in str(expr):
            pre_decls += '(void) post_syn;\n    '
        pre_decls += self.declarations(self.cfun.pre_sum_sym[0], spec)
        code = self.pre_template.format(
            func_pragma=self.func_pragma(spec),
            pre_sum_name=self.kernel_name_pre_sum(i),
            pre_decls=pre_decls,
            pre_sum=expr_c,
            storage=self.storage.value,
            **spec.dict,
        )
        return code  # }}}

    def generate_post(self, i, expr, spec: BaseSpec):  # {{{
        expr_c = self.generate_c(expr, spec)
        post_decls = self.declarations(expr, spec)
        code = self.post_template.format(
            func_pragma=self.func_pragma(spec),
            post_sum_name=self.kernel_name_post_sum(i),
            post_decls=post_decls,
            post_sum=expr_c,
            stat=self.cfun.stat,
            storage=self.storage.value,
            **spec.dict
        )
        return code  # }}}

    def generate_code(self, spec: BaseSpec):
        """
        Generate kernels for each unique pre/post expression.

        """
        self.pre_expr_kname = {}
        self.post_expr_kname = {}
        pre_funcs = []
        for i, expr in enumerate(self.cfun.pre_sum_sym):
            if expr == 0:
                continue
            pre_funcs.append(self.generate_pre(i, expr, spec))
            self.pre_expr_kname[str(expr)] = self.kernel_name_pre_sum(i)
        post_funcs = []
        for i, expr in enumerate(self.cfun.post_sum_sym):
            post_funcs.append(self.generate_post(i, expr, spec))
            self.post_expr_kname[str(expr)] = self.kernel_name_post_sum(i)
        code = self.template.format(
            pre_funcs='\n\n'.join(pre_funcs),
            post_funcs='\n\n'.join(post_funcs),
        )
        return code

    def kernel_name_pre_sum(self, i):
        return 'tvb_%s_pre_sum_%d' % (self.cfun.__class__.__name__, i)

    def kernel_name_post_sum(self, i):
        return 'tvb_%s_post_sum_%d' % (self.cfun.__class__.__name__, i)

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
                value=literal.float_(spec, self.cfun.param[dep.name]),
                **spec.dict
            ))
        return '\n    '.join(lines)
