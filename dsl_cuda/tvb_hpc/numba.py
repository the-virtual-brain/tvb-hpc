import warnings
import logging
import loopy as lp
import loopy.target.numba as base_numba


LOG = logging.getLogger('tvb_hpc')


class NumbaTarget(base_numba.NumbaTarget):

    def __init__(self, *args, **kwargs):
        self.no_jit = kwargs.pop('no_jit', False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            super().__init__(*args, **kwargs)

    def get_kernel_executor_cache_key(self, *args, **kwargs):
        return 'default'

    def get_kernel_executor(self, knl, *args, **kwargs):
        code, _ = lp.generate_code(knl)
        if self.no_jit:
            code = '\n'.join([
                line for line in code.split('\n')
                if line != '@_lpy_numba.jit'])
            for i, line in enumerate(code.split('\n')):
                print(i+1, line)
        LOG.debug(code)
        ns = {}
        exec(code, ns)
        return ns[knl.name]


class NumbaCudaTarget(base_numba.NumbaCudaTarget):

    def get_kernel_executor_cache_key(self, *args, **kwargs):
        return 'default'

    def get_kernel_executor(self, knl, *args, **kwargs):
        code, _ = lp.generate_code(knl)
        LOG.debug(code)
        ns = {}
        exec(code, ns)
        return self._wrap_dims(ns[knl.name])

    def _wrap_dims(self, nbc_knl):
        def _(griddim, blockdim, *args, **kwargs):
            return nbc_knl[griddim, blockdim](*args, **kwargs)
        return _
