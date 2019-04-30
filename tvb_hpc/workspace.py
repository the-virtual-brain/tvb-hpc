import loopy as lp
import pymbolic as pm
import numpy as np
from .utils import getLogger

LOG = getLogger('tvb_hpc')


class Workspace:
    """
    Build workspace for kernel, e.g.

    >>> knl = lp.make_kernel(...)
    >>> wrk = Workspace(knl, ...)
    >>> wrk.data['rand'][:] = np.random.rand(...)
    >>> wrk.rand[:] = np.random.rand(...)
    >>> knl(cq, **wrk.data)

    """

    def __init__(self, kernel: lp.LoopKernel, **scalars):
        self.kernel = kernel
        self.scalars = scalars
        self.data = {}
        for arg in self.kernel.args:
            if isinstance(arg, lp.GlobalArg):
                shape = tuple(self._shape(arg.shape))
                self.data[arg.name] = self._alloc(shape, arg.dtype)
            elif isinstance(arg, lp.ValueArg):
                npdt = getattr(np, arg.dtype.numpy_dtype.name)
                self.data[arg.name] = npdt(scalars[arg.name])

    def _alloc(self, shape, dtype):
        return np.zeros(shape, dtype)

    def _shape(self, shape):
        for dim in shape:
            if isinstance(dim, pm.primitives.Variable):
                name = dim.name
                dim = self.scalars[name]
                self.data[name] = dim
            yield dim


class CLWorkspace(Workspace):

    def __init__(self, cq, *args, **kwargs):
        self.cq = cq
        super().__init__(*args, **kwargs)

    def _alloc(self, shape, dtype):
        from pyopencl.array import zeros
        return zeros(self.cq, shape, dtype)
