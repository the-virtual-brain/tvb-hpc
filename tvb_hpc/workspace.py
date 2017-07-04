import loopy as lp
import pymbolic as pm
import numpy as np
from .utils import getLogger

LOG = getLogger('tvb_hpc')


class Workspace:
    """
    Build workspace for kernel.

    """

    def __init__(self, kernel: lp.LoopKernel, **dim_vals):
        self.kernel = kernel
        self.dim_vals = dim_vals
        self.data = {}
        for arg in self.kernel.args:
            if isinstance(arg, lp.GlobalArg):
                shape = tuple(self._shape(arg.shape))
                self.data[arg.name] = self._alloc(shape, arg.dtype)

    def _alloc(self, shape, dtype):
        return np.zeros(shape, dtype)

    def _shape(self, shape):
        for dim in shape:
            if isinstance(dim, pm.primitives.Variable):
                dim = self.dim_vals[dim.name]
            yield dim


class CLWorkspace(Workspace):

    def __init__(self, cq, *args, **kwargs):
        self.cq = cq
        super().__init__(*args, **kwargs)

    def _alloc(self, shape, dtype):
        from pyopencl.array import zeros
        return zeros(self.cq, shape, dtype)
