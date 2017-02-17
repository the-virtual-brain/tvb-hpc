
"""
The coupling module describes features of coupling functions.

NB bug/feature wrt. TVB here: we assume pre_syn is observables, not state vars.

If single expression given, applied to all observables. Otherwise,
empty expressions stop evaluation of coupling on one or more observables.

A connection is specified node to node, so applies to all cvars.

"""

import numpy as np
import pymbolic as pm
from tvb_hpc.utils import exprs


class BaseCoupling:
    param = {}
    pre_sum = ''
    post_sum = ''

    def __init__(self, model):
        self.model = model
        self.param_sym = np.array([pm.var(name) for name in self.param.keys()])
        self.pre_sum_sym = exprs(self.pre_sum)
        self.post_sum_sym = exprs(self.post_sum)

    @property
    def stat(self):
        # TODO replace by dep analysis
        if 'mean' in self.post_sum[0]:
            return 'mean'
        if 'sum' in self.post_sum[0]:
            return 'sum'
        raise AttributeError('unknown stat in %r' % (self.post_sum, ))


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
