from pymbolic import parse
from pymbolic.mapper.stringifier import SimplifyingSortingStringifyMapper
from sympy.parsing.sympy_parser import parse_expr


def simplify(expr):
    "Simplify pymbolic expr via SymPy."
    # TODO switch entirely to SymPy?
    parts = SimplifyingSortingStringifyMapper()(expr)
    simplified = parse_expr(parts).simplify()
    return parse(repr(simplified))
