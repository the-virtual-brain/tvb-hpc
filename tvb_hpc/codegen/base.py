import enum
import ctypes as ct
from typing import List
from collections import namedtuple
import numpy as np
from pymbolic.mapper.c_code import CCodeMapper


class TypeTable:

    Type = namedtuple('Type', 'name ctypes numpy')

    type_table = [
        Type('float', ct.c_float, np.float32),
        Type('double', ct.c_double, np.float64),
        Type('int', ct.c_int, np.intc),
        Type('unsigned int', ct.c_uint, np.uintc),
        Type('void', ct.c_void_p, np.void),
    ]

    def find_type(self, type):
        """
        Finds correspondences among C type names (str), ctypes types and numpy
        types.

        """
        if isinstance(type, str):
            type = type.strip()
        for types in self.type_table:
            if type in types:
                return types
        msg = 'Unknown type %r'
        raise ValueError(msg % type)


def indent(code, n=4):
    lines = []
    for line in code.split('\n'):
        if line.strip():
            line = n*' ' + line
        lines.append(line)
    return '\n'.join(lines)


class GeneratesC:
    def generate_c(self, spec):
        pass


class Storage(enum.Enum):
    none = ''
    default = ''
    static = 'static'
    inline = 'inline'
    extern = 'extern'


class Unit(GeneratesC):

    template = """
{includes}

{body}
"""

    def __init__(self, body, *includes):
        self.includes = '\n'.join(
            ['#include "%s"' % (inc, ) for inc in includes])
        if isinstance(body, GeneratesC):
            body = body.generate_c()
        self.body = body

    def generate_c(self, spec):
        return self.template.format(**self.__dict__)


class Block(GeneratesC):

    template = """
{{
{body}
}}
"""

    def __init__(self, *args):
        self.args = args

    def generate_c(self, spec):
        body = []
        for arg in self.args:
            if not isinstance(arg, str):
                arg = arg.generate_c(spec)
            body.append(arg)
        body = '\n'.join(body)
        return self.template.format(body=indent(body))


class Func(GeneratesC):

    template = """
{pragma}
{storage}
{restype} {name}({args}) {body_code}
"""

    def __init__(self, name, body, args,
                 restype='void',
                 pragma='',
                 storage=Storage.none,
                 ):
        self.name = name
        if not isinstance(body, Block):
            body = Block(body)
        self.body = body
        self.pragma = pragma
        self.storage = storage.value
        self.restype = restype
        self.args = args
        self.parse_args()

    def generate_c(self, spec):
        body_code = self.body.generate_c(spec)
        return self.template.format(body_code=body_code, **self.__dict__)

    def parse_args(self):
        args = [arg.strip() for arg in self.args.split(',')]
        ttable = TypeTable()
        types, names = [], []
        for arg in args:
            parts = arg.split('*')
            if len(parts) == 2:
                type = ct.POINTER(ttable.find_type(parts[0]).ctypes)
                name = parts[1]
            elif len(parts) == 1:
                type, name = arg.split(' ')
            else:
                msg = 'Not smart enough for %r.'
                raise ValueError(msg % (arg, ))
            types.append(type)
            names.append(type)
        self.types = types
        self.names = names

    def compile(self, compiler, spec):
        self.lib = compiler(self.name, self.generate_c(spec))
        ttable = TypeTable()
        fn = getattr(self.lib, self.name)
        fn.restype = ttable.find_type(self.restype).ctypes
        if self.restype == 'void':
            fn.restype = None
        fn.argtypes = self.types
        self.fn = fn
        return fn

    def __call__(self, *args):
        if not hasattr(self, 'fn'):
            from ..compiler import Compiler
            from .base import BaseSpec
            self.compile(Compiler(), BaseSpec())
        prepped_args = []
        for arg, ctype in zip(args, self.fn.argtypes):
            if hasattr(arg, 'ctypes'):
                arg = arg.ctypes.data_as(ctype)
            else:
                arg = ctype(arg)
            prepped_args.append(arg)
        return self.fn(*prepped_args)


class Loop(GeneratesC):
    template = """
{pragma}
for ({type} {var} = {lo}; {var} < {hi}; {var}++)
{body_code}
"""

    def __init__(self, var, hi, body, lo=0, pragma='', type='unsigned int'):
        self.var = var
        self.lo = lo
        self.hi = hi
        if not isinstance(body, Block):
            body = Block(body)
        self.body = body
        self.pragma = pragma
        self.type = type

    def generate_c(self, spec):
        body_code = indent(self.body.generate_c(spec))
        return self.template.format(body_code=body_code, **self.__dict__)


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
    # see GeneratesC


class BaseLayout:
    """
    Handle layout of data in memory, generating indexing expressions, etc.

    For now, kernels manually / hard code data chunking.

    """

    idx_expr_template = (
            "{i}/{width}*{width}*{nvar} + {j}*{width} + {i}%{width}")

    def __init__(self, width, nvar=None):
        self.width = width
        self.nvar = nvar

    def generate_idx_expr(self, ivar: str, jvar: str, nvar=None):
        nvar = nvar or self.nvar
        assert nvar is not None
        return self.idx_expr_template.format(
            i=ivar, j=jvar, nvar=nvar, width=self.width)


class BaseSpec:
    """
    Spec handles details dtype, vectorization, alignment, etc which affect
    code generation but not the math.

    """

    def __init__(self, float='float', width=8, align=None, openmp=False,
                 layout=None):
        self.float = float
        self.width = width
        self.align = align
        self.openmp = openmp
        self.layout = layout or BaseLayout(width=width)

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
