import numpy as np
from tvb_hpc.model import BaseModel
from tvb_hpc.coupling import BaseCoupling


class DenseNetwork:
    """
    Simple dense weights network, no delays etc.

    """

    def __init__(self, model: BaseModel, cfun: BaseCoupling):
        self.model = model
        self.cfun = cfun

    def _npeval_mat(self, a):
        if a.shape[1] == 0:
            return []
        return np.transpose(a, (1, 0, 2)).reshape((a.shape[1], -1))

    def npeval(self, weights, obsrv, input):
        """
        Evaluate network (obsrv -> weights*cfun -> input) on arrays
        using NumPy.

        """
        # TODO generalize layout.. xarray?
        nn, _, w = obsrv.shape
        ns = {}
        for key in dir(np):
            ns[key] = getattr(np, key)
        ns.update(self.cfun.param)
        obsmat = self._npeval_mat(obsrv)
        for i, (_, pre, post, _) in enumerate(self.cfun.io):
            ns['pre_syn'] = obsmat[i]
            ns['post_syn'] = obsmat[i].reshape((-1, 1))
            weighted = eval(str(pre), ns) * weights
            ns[self.cfun.stat] = getattr(weighted, self.cfun.stat)(axis=1)
            input[:, i, :] = eval(str(post), ns).reshape((nn, w))
