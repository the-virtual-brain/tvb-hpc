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

import enum
from typing import List, Union
import numpy as np
import pymbolic as pm
from pymbolic.mapper.dependency import DependencyMapper
from .utils import exprs
from .model import BaseModel
from .utils import getLogger


# list of Loopy instructions
Isns = List[str]


class PostSumStat(enum.Enum):
    sum = 'sum'
    mean = 'mean'


class BaseCoupling:
    """
    A coupling function describe pre- and post-summation expressions
    which compute the coupling from model observables to mode inputs,
    weighted by connectivity.

    This class is only concerned with describing those functions, not
    producing a kernel which employs these expressions.  For this, please see
    the network classes.

    """

    param = {}
    pre_sum = ''
    post_sum = ''

    def __init__(self, model: BaseModel):
        self.model = model
        self.param_sym = np.array([pm.var(name) for name in self.param.keys()])
        self.pre_sum_sym = exprs(self.pre_sum)
        self.post_sum_sym = exprs(self.post_sum)
        lname = '%s<%s>'
        lname %= self.__class__.__name__, model.__class__.__name__
        self.logger = getLogger(lname)
        self._check_io()

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

    def post_stat(self, i: int) -> PostSumStat:
        """
        Post-summation expressions can refer to 'sum' or 'mean' special
        names referring to the summation or its average.
        This function looks at variables used in the i'th post-summation
        expression and returns 'sum' if found, 'mean' if found, otherwise
        raises an exception.
        """
        mapper = DependencyMapper(include_calls=False)
        dep_names = [dep.name for dep in mapper(self.post_sum_sym[i])]
        if 'mean' in dep_names:
            return PostSumStat('mean')
        if 'sum' in dep_names:
            return PostSumStat('sum')
        raise ValueError('unknown stat in %r' % (self.post_sum, ))


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
