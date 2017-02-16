
"""
Handle code generation tasks. Likely to become a module with
task specific modules.

"""

import numpy as np
import ctypes as ct
from pymbolic.mapper.c_code import CCodeMapper
from typing import List, Dict


class BaseCodeGen:

    def generate_alignments(self, names: List[str], spec: Dict[str, str]):
        value = spec['align']
        lines = []
        for name in names:
            fmt = '{name} = __builtin_assume_aligned({name}, {value});'
            line = fmt.format(name=name, value=value)
            lines.append(line)
        return lines

    def generate_c(self, expr):
        return CCodeMapper()(expr)


class BaseSpec:
    """
    Spec handles details dtype, vectorization, alignment, etc which affect
    code generation but not the math.

    """

    def __init__(self, float='float', width=8, align=64):
        self.float = float
        self.width = width
        self.align = align

    @property
    def dtype(self):
        return self.float

    @property
    def np_dtype(self):
        return {'float': np.float32}[self.dtype]

    @property
    def ct_dtype(self):
        return {'float': ct.c_float}[self.dtype]

    @property
    def dict(self):
        return {
            'float': self.float,
            'width': self.width,
            'align': self.align
        }
