import numpy as np
import pymbolic as pm
from pymbolic.mapper.c_code import CCodeMapper
from pymbolic.mapper.differentiator import DifferentiationMapper
from tvb_hpc.utils import simplify



def vars(svars):
    return np.array([pm.var(var) for var in svars.split()])


def exprs(sexprs):
    exprs = []
    for expr in sexprs:
        if isinstance(expr, (int, float)):
            exprs.append(expr)
        else:
            try:
                exprs.append(pm.parse(expr))
            except Exception as exc:
                raise Exception(repr(expr))
    return np.array(exprs)


class BaseModel:
    """
    BaseModel parses attributes on subclasses, defining a networked SDE.

    """

    state = ''
    input = ''
    param = ''
    drift = '',
    diffs = '',
    obsrv = '',
    const = {}
    auxex = []

    def __init__(self):
        self.state_sym = vars(self.state)
        self.input_sym = vars(self.input)
        self.param_sym = vars(self.param)
        self.drift_sym = exprs(self.drift)
        self.diffs_sym = exprs(self.diffs)
        self.obsrv_sym = exprs(self.obsrv)

    @property
    def indvars(self):
        return np.r_[self.state_sym, self.input_sym, self.param_sym]

    def partial(self, expr):
        exprs = []
        for var in self.indvars:
            exprs.append(simplify(DifferentiationMapper(var)(expr)))
        return np.array(exprs)

    def declarations(self, spec):
        common = {'nsvar': len(self.state_sym), }
        common.update(spec)
        lines = []
        # add constants
        for name, value in self.const.items():
            fmt = '{float} {name} = (({float}) {value});'
            line = fmt.format(name=name, value=value, **common)
            lines.append(line)
        return lines

    def inner_loop_lines(self, spec):
        common = {
            'nsvar': len(self.state_sym),
            'nsvar_width': len(self.state_sym) * spec['width']
        }
        common.update(spec)
        lines = []
        # unpack state, input & parameters
        fmt = '{float} {var} = {kind}[i * {nsvar_width} + {isvar} * {width} + j];'
        for kind in 'state input param'.split():
            vars = getattr(self, kind + '_sym')
            for i, var in enumerate(vars):
                line = fmt.format(
                    kind=kind, var=var.name, isvar=i, **common)
                lines.append(line)
        # do aux exprs
        fmt = '{float} {lhs} = {rhs};'
        for lhs, rhs in self.auxex:
            line = fmt.format(lhs=lhs, rhs=rhs, **common)
            lines.append(line)
        # store drift / diffs / obsrv
        fmt = '{kind}[i * {nsvar_width} + {isvar} * {width} + j] = {expr};'
        for kind in 'drift diffs obsrv'.split():
            exprs = getattr(self, kind + '_sym')
            for i, expr in enumerate(exprs):
                line = fmt.format(
                    kind=kind, expr=self._cc(expr), isvar=i, **common)
                lines.append(line)
        return lines

    @property
    def kernel_name(self):
        return 'tvb_' + self.__class__.__name__

    def _cc(self, expr):
        return CCodeMapper()(expr)

    def prep_arrays(self, nnode, spec):
        dtype = {'float': np.float32, 'double': np.float64}[spec['float']]
        arrs = []
        for key in 'state input param drift diffs obsrv'.split():
            shape = nnode, len(getattr(self, key + '_sym')), spec['width']
            arrs.append(np.zeros(shape, dtype))
        return arrs


class test_model(BaseModel):
    state = 'y1 y2'
    input = 'i1'
    param = 'a b c'
    drift = '(y1 - y1**3/3 + y2)*b', '(a - y1 + i1)/b'
    diffs = 'c', 'c'
    obsrv = 'y1',
    const = {'d': 3.0, 'e': -12.23904e-2}


class Kuramoto(BaseModel):
    state = 'theta'
    state_limits = {'theta': (-np.pi, np.pi, 'wrap')}
    input = 'I'
    param = 'omega'
    drift = 'omega + I'
    obsrv = 'sin(theta)',


class HMJE(BaseModel):
    state = 'x1 y1 z x2 y2 g'
    input = 'c1 c2'
    param = 'x0 Iext r'
    drift = (
        'tt * (y1 - z + Iext + Kvf * c1 + ('
                '  (x1 <  0)*(-a * x1 * x1 + b * x1)'
                '+ (x1 >= 0)*(slope - x2 + 0.6 * (z - 4)**2)'
            ') * x1)',
        'tt * (c - d * x1 * x1 - y1)',
        'tt * (r * (4 * (x1 - x0) - z + (z < 0) * (-0.1 * pow(z, 7)) + Ks * c1))',
        'tt * (-y2 + x2 - x2*x2*x2 + Iext2 + 2 * g - 0.3 * (z - 3.5) + Kf * c2)',
        'tt * ((-y2 + (x2 >= (-3.5)) * (aa * (x2 + 0.25))) / tau)',
        'tt * (-0.01 * (g - 0.1 * x1))'
    )
    diffs = 0, 0, 0, 0.0003, 0.0003, 0
    obsrv = '-x1 + x2', 'z'
    const = {'Iext2': 0.45, 'a': 1, 'b': 3, 'slope': 0, 'tt': 1, 'c': 1,
             'd': 5, 'Kvf': 0, 'Ks': 0, 'Kf': 0, 'aa': 6, 'tau': 10,
             'x0': -1.6, 'Iext': 3.1, 'r': 0.00035}


class RWW(BaseModel):
    state = 'S'
    state_limits = {'S': (0, 1, 'clip')}
    input = 'c'
    param = 'w io'
    const = {'a': 0.270, 'b': 0.108, 'd': 154.0, 'g': 0.641,
             'ts': 100.0, 'j': 0.2609, 'w': 0.6, 'io': 0.33}
    auxex = [
        ('x', 'w * j * S + io + j * c'),
        ('h', '(a * x - b) / (1 - exp(-d*(a*x - b)))')
    ]
    drift = '- (S / ts) + (1 - S) * h * g',
    diffs = 0.01,
    obsrv = 'S',
