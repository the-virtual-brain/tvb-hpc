import ctypes as ct
from typing import List, Dict

import numpy as np
from pymbolic.mapper.c_code import CCodeMapper


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

    def generate_alignments(self, names: List[str], spec: Dict[str, str]):
        """
        Generates a series of statement lines which declare a pointer to be
        aligned to the alignment given by ``spec``.

        >>> BaseCodeGen().generate_alignments(['a', 'b'], {'align': 64})
        ['a = __builtin_assume_aligned(a, 64);', ...]

        """
        value = spec['align']
        lines = []
        if value is None:
            return lines
        for name in names:
            fmt = '{name} = __builtin_assume_aligned({name}, {value});'
            line = fmt.format(name=name, value=value)
            lines.append(line)
        return lines

    def generate_c(self, expr):
        """
        Generate C code for `expr` argument.
        """
        return CCodeMapper()(expr)


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

    # TODO