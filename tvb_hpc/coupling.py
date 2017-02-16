
"""
bug/feature wrt. TVB here: we assume pre_syn is observables, not state vars.

If single expression given, applied to all observables. Otherwise,
empty expressions stop evaluation of coupling on one or more observables.

A connection is specified node to node, so applies to all cvars.

"""


from tvb_hpc.utils import exprs


class BaseCoupling:
    def __init__(self, model):
        self.model = model
        # TODO parse descriptions & generate core


class Linear(BaseCoupling):
    param = 'a b'
    const = {'a': 1e-3, 'b': 0}
    pre_sum = 'pre_syn'
    post_sum = 'a * sum + b'


class Sigmoidal(BaseCoupling):
    param = 'cmin cmax midpoint r a'
    pre_sum = 'cmax / (1 + exp(r * (midpoint - pre_syn)))'
    post_sum = 'a * sum'


class Diff(Linear):
    pre_sum = 'pre_syn - post_syn'


class Kuramoto(Diff):
    pre_sum = 'sin(pre_syn - post_syn)', ''
    post_sum = 'a * mean'
