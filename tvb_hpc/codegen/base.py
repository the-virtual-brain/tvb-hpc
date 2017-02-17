import enum
import ctypes as ct
from typing import List

import numpy as np
from pymbolic.mapper.c_code import CCodeMapper


class Storage(enum.Enum):
    static = 'static'
    inline = 'inline'
    extern = 'extern'


class MyCCodeMapper(CCodeMapper):

    _float_funcs = {
        f: f + 'f'
        for f in 'sin cos exp pow'.split()
    }

    def __init__(self, *args, **kwargs):
        self.use_float = kwargs.pop('use_float', True)
        super().__init__(*args, **kwargs)

    # lifted from pymbolic, slight modified
    def map_call(self, expr, enclosing_prec):
        from pymbolic.primitives import Variable
        from pymbolic.mapper.stringifier import PREC_NONE, PREC_CALL
        if isinstance(expr.function, Variable):
            func = expr.function.name
        else:
            func = self.rec(expr.function, PREC_CALL)
        func = self._float_funcs.get(func, func)
        return self.format(
                "%s(%s)",
                func, self.join_rec(", ", expr.parameters, PREC_NONE))

    def map_power(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        from pymbolic.primitives import is_constant, is_zero
        if is_constant(expr.exponent):
            if is_zero(expr.exponent):
                return "1"
            elif is_zero(expr.exponent - 1):
                return self.rec(expr.base, enclosing_prec)
            elif is_zero(expr.exponent - 2):
                return self.rec(expr.base*expr.base, enclosing_prec)
            # cube is common in our code
            elif is_zero(expr.exponent - 3):
                return self.rec(expr.base*expr.base*expr.base, enclosing_prec)
        return self.format(
                "pow(%s, %s)",
                self.rec(expr.base, PREC_NONE),
                self.rec(expr.exponent, PREC_NONE))


class BaseCodeGen:

    # http://www.cplusplus.com/reference/cmath/
    _math_names = (
        'cos sin tan acos asin atan atan2 cosh sinh tanh acosh asinh atanh '
        'exp frexp ldexp log log10 modf exp2 expm1 ilogb log1p log2 logb '
        'scalbn scalbln pow sqrt cbrt hypot erf erfc tgamma lgamma ceil '
        'floor fmod trunc round lround llround rint lrint llrint nearbyint '
        'remainder remquo copysign nan nextafter nexttoward fdim fmax fmin '
        'fabs abs fma fpclassify isfinite isinf isnan isnormal signbit '
        'isgreater isgreaterequal isless islessequal islessgreater '
        'isunordered INFINITY NAN HUGE_VAL HUGE_VALF HUGE_VALL '
    ).split(' ')

    @property
    def math_names(self):
        """
        A list of names of mathematical functions which can be
        expected to be available. By default, it includes all names provided by
        the standard C/C++ math library, but maybe more limited for certain
        targets like CUDA.

        """
        return self._math_names

    def generate_alignments(self, names: List[str], spec: 'BaseSpec'):
        """
        Generates a series of statement lines which declare a pointer to be
        aligned to the alignment given by ``spec``.

        >>> BaseCodeGen().generate_alignments(['a', 'b'], {'align': 64})
        ['a = __builtin_assume_aligned(a, 64);', ...]

        """
        value = spec.align
        lines = []
        if value is None:
            return lines
        for name in names:
            fmt = '{name} = __builtin_assume_aligned({name}, {value});'
            line = fmt.format(name=name, value=value)
            lines.append(line)
        return lines

    def generate_c(self, expr, spec: 'BaseSpec'):
        """
        Generate C code for `expr` argument.
        """
        mapper = MyCCodeMapper(use_float=spec.float == 'float')
        return mapper(expr)

    # TODO common interface? kernel_name, decls, innerloop, pragma etc


class BaseSpec:
    """
    Spec handles details dtype, vectorization, alignment, etc which affect
    code generation but not the math.

    """

    def __init__(self, float='float', width=8, align=None, openmp=False):
        self.float = float
        self.width = width
        self.align = align
        self.openmp = False

    @property
    def dtype(self):
        return self.float

    @property
    def np_dtype(self):
        return {'float': np.float32}[self.dtype]

    @property
    def ct_dtype(self):
        return {'float': ct.c_float}[self.dtype]

    # TODO refactor so not needed
    @property
    def dict(self):
        return {
            'float': self.float,
            'width': self.width,
            'align': self.align,
            'openmp': self.openmp
        }


class BaseLayout:
    """
    Handle layout of data in memory, generating indexing expressions, etc.

    For now, kernels manually / hard code data chunking.

    """

    idx_expr_template = (
            "{i}/{width}*{width}*{nvar} + {j}*{width} + {i}%{width}")

    def __init__(self, nvar, width):
        self.nvar = nvar
        self.width = width

    def generate_idx_expr(self, ivar: str, jvar: str):
        return self.idx_expr_template.format(
            i=ivar,
            j=jvar,
            nvar=self.nvar,
            width=self.width)
