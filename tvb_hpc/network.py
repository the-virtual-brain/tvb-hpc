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
import loopy as lp
import pymbolic as pm
from .base import BaseKernel
from .compiler import Spec
from .model import BaseModel
from .coupling import BaseCoupling, PostSumStat
from .utils import getLogger, subst_vars


LOG = getLogger(__name__)


class Network(BaseKernel):
    """
    Network implementing spare time delay coupling with pre/post summation
    coupling functions,

      input[i, k] = cfpost[k]( sum(j, w[i,j]*cfpre[k](obsrv[j,k], obsrv[i,k])))

    In TVB, we use circular indexing, but this incurs penalty in terms of
    complexity of data structures, always performing integer modulo, etc.  Far
    simpler to assume contiguous time and periodically reconstruct as memory
    constraints require, which can be done in a background thread to minimize
    overhead.

    """

    def __init__(self, model: BaseModel, cfun: BaseCoupling):
        self.model = model
        self.cfun = cfun

    def _insn_cfun(self, k, pre, post):
        "Generates an instruction for a single coupling function."
        # TODO add loopy hints to make more efficient
        # substitute pre_syn and post_syn for obsrv data
        pre_expr = subst_vars(
            expr=pre,
            pre_syn=pm.parse('obsrv[(i_time - delays[j_node]) % ntime, col[j_node], k]'),  # k is var idx
            post_syn=pm.parse('obsrv[i_time % ntime, i_node, k]'),
        )
        # build weighted sum over nodes
        sum = subst_vars(
            expr=pm.parse('sum(j_node, weights[j_node] * pre_expr)'),
            pre_expr=pre_expr,
        )
        # mean used by some cfuns
        mean = sum / pm.var('nnode')
        # subst mean / sum through post cfun
        post_expr = subst_vars(post, sum=sum, mean=mean)
        # generate store instruction for post expr, with params
        post_expr = subst_vars(post_expr, k=k, **self.cfun.param)
        return 'input[i_node, %d] = %s' % ( k, post_expr )

    def kernel_isns(self):
        "Generates instructions for all cfuns"
        yield '<> j_node_lo = row[i_node]'
        yield '<> j_node_hi = row[i_node + 1]'
        for k, (_, pre, post, _) in enumerate(self.cfun.io):
            yield self._insn_cfun(k, pre, post)

    def kernel_domains(self):
        return [
            '{ [i_node]: 0 <= i_node < nnode }',
            '{ [j_node]: j_node_lo <= j_node < j_node_hi }'
        ]

    def kernel_dtypes(self):
        dtypes = {
            'i_time,ntime,nnode,nnz,delays,row,col': np.uintc,
            'input,obsrv,weights': np.float32,
        }
        for name, val in self.cfun.param.items():
            from pymbolic.primitives import Variable
            if isinstance(val, Variable):
                dtypes[name] = np.float32
        return dtypes

    def kernel_data(self):
        data = super().kernel_data()
        # loopy can't infer bound on first dim of obsrv
        shape = pm.var('ntime'), pm.var('nnode'), len(self.cfun.io)
        data[data.index('obsrv')] = lp.GlobalArg('obsrv', shape=shape)
        # nor that of nnz length vectors
        nnz_shape = pm.var('nnz'),
        for key in 'col delays weights'.split():
            data[data.index(key)] = lp.GlobalArg(key, shape=nnz_shape)
        return data
