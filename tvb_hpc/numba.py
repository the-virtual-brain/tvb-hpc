import logging
import loopy as lp


LOG = logging.getLogger(__name__)


class NumbaTarget(lp.target.numba.NumbaTarget):

    def get_kernel_executor_cache_key(self, *args, **kwargs):
        return 'default'

    def get_kernel_executor(self, knl, *args, **kwargs):
        code, _ = lp.generate_code(knl)
        LOG.debug(code)
        ns = {}
        exec(code, ns)
        return ns[knl.name]
