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
The coupling module describes features of coupling functions.

NB bug/feature wrt. TVB here: we assume pre_syn is observables, not state vars.

If single expression given, applied to all observables. Otherwise,
empty expressions stop evaluation of coupling on one or more observables.

A connection is specified node to node, so applies to all cvars.

"""

import numpy as np
import pymbolic as pm
from pymbolic.mapper.dependency import DependencyMapper
from .codegen import BaseCodeGen, Storage
from .compiler import Spec
from .utils import exprs
from .model import BaseModel
from .utils import getLogger


class BaseCoupling(BaseCodeGen):

    param = {}
    pre_sum = ''
    post_sum = ''

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

    def __init__(self, model: BaseModel, storage=Storage.static):
        self.model = model
        self.param_sym = np.array([pm.var(name) for name in self.param.keys()])
        self.pre_sum_sym = exprs(self.pre_sum)
        self.post_sum_sym = exprs(self.post_sum)
        lname = '%s<%s>'
        lname %= self.__class__.__name__, model.__class__.__name__
        self.logger = getLogger(lname)
        self._check_io()
        self.kernel_names = []
        self.storage = storage

    def _check_io(self):
        obsrv_sym = self.model.obsrv_sym
        if self.model.input_sym.size < obsrv_sym.size:
            msg = 'input shorter than obsrv, truncating obsrv used for cfun.'
            self.logger.debug(msg)
            obsrv_sym = obsrv_sym[:self.model.input_sym.size]
        terms = (
            obsrv_sym,
            self.pre_sum_sym,
            self.post_sum_sym,
            self.model.input_sym
        )
        bcast = np.broadcast(*terms)
        self.io = list(bcast)
        fmt = 'io[%d] (%s) -> (%s) -> (%s) -> (%s)'
        for i, parts in enumerate(self.io):
            self.logger.debug(fmt, i, *parts)

    @property
    def stat(self):
        # TODO replace by dep analysis
        if 'mean' in self.post_sum[0]:
            return 'mean'
        if 'sum' in self.post_sum[0]:
            return 'sum'
        raise AttributeError('unknown stat in %r' % (self.post_sum, ))

    def func_pragma(self, spec):
        func_pragma = ''
        if spec.openmp:
            func_pragma = '#pragma omp declare simd'
        return func_pragma

    # TODO use classes for pre/psot functions, so we can have API
    # to ask things like, does it use post_syn value etc..

    def generate_pre(self, i, expr, spec: Spec):
        expr_c = self.generate_c(expr, spec)
        pre_decls = ''
        if 'post_syn' not in str(expr):
            pre_decls += '(void) post_syn;\n    '
        pre_decls += self.declarations(self.pre_sum_sym[0], spec)
        code = self.pre_template.format(
            func_pragma=self.func_pragma(spec),
            pre_sum_name=self.kernel_name_pre_sum(i),
            pre_decls=pre_decls,
            pre_sum=expr_c,
            storage=self.storage.value,
            **spec.dict,
        )
        return code

    def generate_post(self, i, expr, spec: Spec):
        expr_c = self.generate_c(expr, spec)
        post_decls = self.declarations(expr, spec)
        code = self.post_template.format(
            func_pragma=self.func_pragma(spec),
            post_sum_name=self.kernel_name_post_sum(i),
            post_decls=post_decls,
            post_sum=expr_c,
            stat=self.stat,
            storage=self.storage.value,
            **spec.dict
        )
        return code

    def generate_code(self, spec: Spec):
        """
        Generate kernels for each unique pre/post expression.

        """
        self.pre_expr_kname = {}
        self.post_expr_kname = {}
        pre_funcs = []
        for i, expr in enumerate(self.pre_sum_sym):
            if expr == 0:
                continue
            pre_funcs.append(self.generate_pre(i, expr, spec))
            self.pre_expr_kname[str(expr)] = self.kernel_name_pre_sum(i)
        post_funcs = []
        for i, expr in enumerate(self.post_sum_sym):
            post_funcs.append(self.generate_post(i, expr, spec))
            self.post_expr_kname[str(expr)] = self.kernel_name_post_sum(i)
        code = self.template.format(
            pre_funcs='\n\n'.join(pre_funcs),
            post_funcs='\n\n'.join(post_funcs),
        )
        return code

    def kernel_name_pre_sum(self, i):
        return 'tvb_%s_pre_sum_%d' % (self.__class__.__name__, i)

    def kernel_name_post_sum(self, i):
        return 'tvb_%s_post_sum_%d' % (self.__class__.__name__, i)

    def declarations(self, expr, spec):
        lines = []
        mapper = DependencyMapper(include_calls=False)
        for dep in mapper(expr):
            if dep.name in ('pre_syn', 'post_syn', 'mean', 'sum'):
                continue
            if dep.name in self.math_names:
                continue
            fmt = '{float} {name} = (({float}) {value});'
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
