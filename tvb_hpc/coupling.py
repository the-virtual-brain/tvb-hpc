
"""
bug/feature wrt. TVB here: we assume pre_syn is observables, not state vars.

If single expression given, applied to all observables. Otherwise,
empty expressions stop evaluation of coupling on one or more observables.

A connection is specified node to node, so applies to all cvars.

"""

import numpy as np
import pymbolic as pm
from pymbolic.mapper.dependency import DependencyMapper
from tvb_hpc.utils import exprs, vars
from tvb_hpc.codegen import BaseCodeGen


class BaseCoupling(BaseCodeGen):
    param = {}
    pre_sum = ''
    post_sum = ''

    template = """
#include <math.h>

#pragma omp declare simd
void {pre_sum_name}({float} pre_syn, {float} post_syn)
{{
    {pre_decls}
    return {pre_sum};
}}

#pragma omp declare simd
void {post_sum_name}({float} {stat}, {float} * __restrict param)
{{
    {post_decls}
    return {post_sum};
}}
"""

    def __init__(self, model):
        self.model = model
        self.param_sym = np.array([pm.var(name) for name in self.param.keys()])
        self.pre_sum_sym = exprs(self.pre_sum)
        self.post_sum_sym = exprs(self.post_sum)

    def generate_code(self, spec):
        assert len(self.pre_sum_sym) == 1
        assert len(self.post_sum_sym) == 1
        pre_sum = self.generate_c(self.pre_sum_sym[0])
        post_sum = self.generate_c(self.post_sum_sym[0])
        code = self.template.format(
            pre_sum_name=self._pre_sum_name,
            post_sum_name=self._post_sum_name,
            pre_sum=pre_sum,
            post_sum=post_sum,
            pre_decls=self.declarations(self.pre_sum_sym[0], spec),
            post_decls=self.declarations(self.post_sum_sym[0], spec),
            stat=self._stat,
            **spec.dict
        )
        return code

    @property
    def _stat(self):
        # TODO replace by dep analysis
        if 'mean' in self.post_sum[0]:
            return 'mean'
        if 'sum' in self.post_sum[0]:
            return 'sum'
        raise AttributeError('unknown stat in %r' % (self.post_sum, ))

    @property
    def _pre_sum_name(self):
        return 'tvb_%s_pre_sum' % (self.__class__.__name__, )

    @property
    def _post_sum_name(self):
        return 'tvb_%s_post_sum' % (self.__class__.__name__, )

    def declarations(self, expr, spec):
        lines = []
        mapper = DependencyMapper(include_calls=False)
        skip_names = 'sin cos exp pow tanh'.split()
        for dep in mapper(expr):
            if dep.name in ('pre_syn', 'post_syn', 'mean', 'sum'):
                continue
            if dep.name in skip_names:
                continue
            fmt = '{float} {name} = {value};'
            lines.append(fmt.format(
                name=dep.name,
                value=self.param[dep.name],
                **spec.dict
            ))
        return '\n    '.join(lines)


class Linear(BaseCoupling):
    param = {'a': 1e-3, 'b': 0}
    pre_sum = 'pre_syn',
    post_sum = 'a * sum + b',


class Sigmoidal(BaseCoupling):
    param = {'cmin': 0, 'cmax': 0.005, 'midpoint': 6, 'r': 1, 'a': 0.56}
    pre_sum = 'cmax / (1 + exp(r * (midpoint - pre_syn)))',
    post_sum = 'a * sum',


class Diff(Linear):
    pre_sum = 'pre_syn - post_syn',


class Kuramoto(Diff):
    pre_sum = 'sin(pre_syn - post_syn)',
    post_sum = 'a * mean',
