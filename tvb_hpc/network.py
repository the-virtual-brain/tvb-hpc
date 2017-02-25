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
import pymbolic as pm
from .base import BaseKernel
from .compiler import Spec
from .model import BaseModel
from .coupling import BaseCoupling, PostSumStat
from .utils import getLogger, subst_vars


LOG = getLogger(__name__)


class DenseNetwork(BaseKernel):
    """
    Simple dense weights network, no delays.

    """

    def __init__(self, model: BaseModel, cfun: BaseCoupling):
        self.model = model
        self.cfun = cfun

    # _insn_cfun uses these expressions to determine how the presynaptic
    # expression is formed.  It is not inlined so that the delay network
    # subclass can trivially override to implement delays.
    _pre_syn_fmt = 'obsrv[j, k]'
    _post_syn_fmt = 'obsrv[i, k]'

    def _insn_cfun(self, k, pre, post):
        "Generates an instruction for a single coupling function."
        # TODO add loopy hints to make more efficient
        # substitute pre_syn and post_syn for obsrv data
        pre_expr = subst_vars(
            expr=pre,
            pre_syn=pm.parse(self._pre_syn_fmt),  # k -> var idx
            post_syn=pm.parse(self._post_syn_fmt),
        )
        # build weighted sum over nodes
        sum = subst_vars(
            expr=pm.parse('sum(j, weights[i, j] * pre_expr)'),
            pre_expr=pre_expr,
        )
        # mean used by some cfuns
        mean = sum / pm.var('nnode')
        # subst mean / sum through post cfun
        post_expr = subst_vars(post, sum=sum, mean=mean)
        # generate store instruction for post expr, with params
        post_expr = subst_vars(post_expr, k=k, **self.cfun.param)
        return 'input[i, %d] = %s' % ( k, post_expr )

    def kernel_isns(self):
        "Generates instructions for all cfuns"
        for k, (_, pre, post, _) in enumerate(self.cfun.io):
            yield self._insn_cfun(k, pre, post)

    def kernel_domains(self):
        return '{ [i, j]: 0 <= i, j < nnode }'

    def kernel_dtypes(self):
        return {
            'nnode': np.uintc,
            'input': np.float32,
            'obsrv': np.float32,
            'weights': np.float32,
        }


class DelayNetwork(DenseNetwork):
    """
    Subclass of dense network implementing time delays.

    In TVB, we use circular indexing, but this incurs penalty in terms of
    complexity of data structures, always performing integer modulo, etc.  Far
    simpler to assume contiguous time and periodically reconstruct as memory
    constraints require, which can be done in a background thread to minimize
    overhead. This isn't yet implemented, but to be looked at.

    """

    _post_syn_fmt = 'obsrv[t, i, k]'
    _pre_syn_fmt = 'obsrv[t - delays[i, j], j, k]'

    def kernel_data(self):
        data = super().kernel_data()
        # need to reassure loopy that obsrv has a bound on first index
        ncvar = len(self.cfun.io)
        nnode = pm.var('nnode')
        ntime = pm.var('ntime')
        from loopy import GlobalArg
        obsrv = GlobalArg('obsrv', shape=(ntime, nnode, ncvar))
        data[data.index('obsrv')] = obsrv
        return data

    def kernel_dtypes(self):
        dtypes = super().kernel_dtypes()
        dtypes['t'] = np.uintc
        dtypes['ntime'] = np.uintc
        dtypes['delays'] = np.uintc
        return dtypes
