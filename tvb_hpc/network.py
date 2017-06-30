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
from scipy import sparse
import loopy as lp
import pymbolic as pm
from .base import BaseKernel
from .model import BaseModel
from .coupling import BaseCoupling
from .utils import getLogger, subst_vars


LOG = getLogger(__name__)


class Connectivity:
    """
    Data class, managing sparse connectivity

    """

    # TODO params like normalization mode, etc.
    # TODO abstraction on data munging, mapping into kernel workspace
    #      needs a lot of thought
    def __init__(self,
                 nnode: int,
                 nz: int,
                 col: np.ndarray[int],
                 row: np.ndarray[int],
                 wnz: np.ndarray[float],
                 lnz: np.ndarray[float]):
        self.nnode = nnode
        self.nz = nz
        self.col = col
        self.row = row
        self.wnz = wnz
        self.lnz = lnz

    @classmethod
    def from_dense(cls, weights, lengths):
        nnode = weights.shape[0]
        nz = ~(weights == 0)
        nnz = nz.sum()
        wnz = weights[nz]
        lnz = lengths[nz]
        sw = sparse.csr_matrix(weights)
        col = sw.indices.astype(np.uintc)
        row = sw.indptr.astype(np.uintc)
        obj = cls(nnode, nnz, col, row, wnz, lnz)
        obj.weights = weights
        obj.lengths = lengths

    @classmethod
    def from_npz(cls,
                 fname,
                 weights_key='weights',
                 lengths_key='lengths'
                 ):
        npz = np.load(fname)
        weights = npz[weights_key]
        lengths = npz[lengths_key]
        return cls.from_dense(weights, lengths)

    @classmethod
    def hcp0(cls):
        return cls.from_npz('data/hcp0.npz')


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
            #                                                     k is var idx
            pre_syn=pm.parse('obsrv[i_time - delays[j_node], col[j_node], k]'),
            post_syn=pm.parse('obsrv[i_time, i_node, k]'),
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
        return 'input[i_node, %d] = %s' % (k, post_expr)

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
