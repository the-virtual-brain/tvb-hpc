import logging
import loopy as lp
import loopy.target.numba as base_numba
import loopy.target.python as base_python
from loopy.diagnostic import LoopyError


LOG = logging.getLogger(__name__)


class ExpressionToPythonMapper(base_python.ExpressionToPythonMapper):

    def map_group_hw_index(self, expr, enclosing_prec):
        if expr.axis != 0:
            raise LoopyError('Only g.0 supported.')
        return expr.name

    def map_local_hw_index(self, expr, enclosing_prec):
        raise LoopyError('Local work tag not supported.')


class NumbaJITASTBuilder(base_numba.NumbaJITASTBuilder):

    def get_expression_to_code_mapper(self, codegen_state):
        return ExpressionToPythonMapper(codegen_state)


class NumbaTarget(base_numba.NumbaTarget):

    def get_kernel_executor_cache_key(self, *args, **kwargs):
        return 'default'

    def get_kernel_executor(self, knl, *args, **kwargs):
        code, _ = lp.generate_code(knl)
        LOG.debug(code)
        ns = {}
        exec(code, ns)
        return ns[knl.name]

    def get_device_ast_builder(self):
        return NumbaJITASTBuilder(self)