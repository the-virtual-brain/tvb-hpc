import logging
import loopy as lp
import loopy.target.numba as base_numba


LOG = logging.getLogger(__name__)


class NumbaTarget(base_numba.NumbaTarget):

    def get_kernel_executor_cache_key(self, *args, **kwargs):
        return 'default'

    def get_kernel_executor(self, knl, *args, **kwargs):
        code, _ = lp.generate_code(knl)
        LOG.debug(code)
        ns = {}
        exec(code, ns)
        return ns[knl.name]
