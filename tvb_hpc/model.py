import numpy as np
import pymbolic as pm
from pymbolic.mapper.differentiator import DifferentiationMapper
from tvb_hpc.codegen import BaseCodeGen
from tvb_hpc.utils import simplify, vars, exprs


class BaseModel(BaseCodeGen):
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

    template = """
#include <math.h>

void {name}(
    unsigned int nnode,
    {float} * __restrict state,
    {float} * __restrict input,
    {float} * __restrict param,
    {float} * __restrict drift,
    {float} * __restrict diffs,
    {float} * __restrict obsrv
)
{{
  {decls}
  {loop_pragma}
  for (unsigned int j=0; j < nnode; j++)
  {{
    unsigned int i = j / {width};
    {body}
  }}
}}
"""

    def generate_code(self, spec):
        decls = self.generate_alignments(
            'state input param drift diffs obsrv'.split(), spec)
        decls += self.declarations(spec)
        body = self.inner_loop_lines(spec)
        code = self.template.format(
            decls='\n  '.join(decls),
            body='\n    '.join(body),
            name=self.kernel_name,
            nsvar=len(self.state_sym),
            loop_pragma='#pragma omp simd safelen(%d)' % (spec['width'], ),
            **spec
        )
        if spec['float'] == 'float':
            code = code.replace('pow', 'powf')
        return code

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
            rhs = self.generate_c(pm.parse(rhs))
            line = fmt.format(lhs=lhs, rhs=rhs, **common)
            lines.append(line)
        # store drift / diffs / obsrv
        fmt = '{kind}[i * {nsvar_width} + {isvar} * {width} + j] = {expr};'
        for kind in 'drift diffs obsrv'.split():
            exprs = getattr(self, kind + '_sym')
            for i, expr in enumerate(exprs):
                line = fmt.format(
                    kind=kind, expr=self.generate_c(expr), isvar=i, **common)
                lines.append(line)
        return lines

    @property
    def kernel_name(self):
        return 'tvb_' + self.__class__.__name__

    def prep_arrays(self, nnode, spec):
        dtype = {'float': np.float32, 'double': np.float64}[spec['float']]
        arrs = []
        for key in 'state input param drift diffs obsrv'.split():
            shape = nnode, len(getattr(self, key + '_sym')), spec['width']
            arrs.append(np.zeros(shape, dtype))
        return arrs


class _TestModel(BaseModel):
    state = 'y1 y2'
    input = 'i1'
    param = 'a b c'
    auxex = [('y1_3', 'y1 * y1 * y1')]
    drift = '(y1 - y1_3/3 + y2)*b', '(a - y1 + i1)/b'
    diffs = 'c', 'c'
    obsrv = 'y1',
    const = {'d': 3.0, 'e': -12.23904e-2}


class Kuramoto(BaseModel):
    "Kuramoto model of phase synchronization."
    state = 'theta'
    state_limits = {'theta': (-np.pi, np.pi, 'wrap')}
    input = 'I'
    param = 'omega'
    drift = 'omega + I',
    diffs = 0,
    obsrv = 'theta', 'sin(theta)'


class HMJE(BaseModel):
    "Hindmarsh-Rose-Jirsa Epileptor model of seizure dynamics."
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
    obsrv = 'x1', 'x2', 'z', '-x1 + x2'
    const = {'Iext2': 0.45, 'a': 1, 'b': 3, 'slope': 0, 'tt': 1, 'c': 1,
             'd': 5, 'Kvf': 0, 'Ks': 0, 'Kf': 0, 'aa': 6, 'tau': 10,
             'x0': -1.6, 'Iext': 3.1, 'r': 0.00035}


class RWW(BaseModel):
    "Reduced Wong-Wang firing rate model."
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


class JansenRit(BaseModel):
    "Jansen-Rit model of visual evoked potentials."
    state = 'y0 y1 y2 y3 y4 y5'
    const = {'A': 3.25, 'B': 22.0, 'a': 0.1, 'b': 0.05, 'v0': 5.52,
             'nu_max': 0.0025, 'r': 0.56, 'J': 135.0, 'a_1': 1.0, 'a_2': 0.8,
             'a_3': 0.25, 'a_4': 0.25, 'p_min': 0.12, 'p_max': 0.32, 'mu': 0.22}
    input = 'lrc'
    param = 'v0 r'
    auxex = [
        ('sigm_y1_y2', '2 * nu_max / (1 + exp(r * (v0 - (y1 - y2))))'),
        ('sigm_y0_1', '2 * nu_max / (1 + exp(r * (v0 - (a_1 * J * y0))))'),
        ('sigm_y0_3', '2 * nu_max / (1 + exp(r * (v0 - (a_3 * J * y0))))'),
    ]
    drift = (
        'y3', 'y4', 'y5',
        'A * a * sigm_y1_y2 - 2 * a * y3 - a**2 * y0',
        'A * a * (mu + a_2 * J * sigm_y0_1 + lrc) - 2 * a * y4 - a**2 * y1',
        'B * b * (a_4 * J * sigm_y0_3) - 2 * b * y5 - b**2 * y2'
     )
    diffs = 0, 0, 0, 0, 0, 0
    obsrv = 'y0 - y1',  # TODO check w/ Andreas


class Linear(BaseModel):
    'Linear differential equation'
    state = 'x'
    input = 'c'
    param = 'lambda'
    const = {'lambda': -1}  # default value
    drift = 'lambda * x',
    diffs = 1e-2,
    obsrv = 'x',


class G2DO(BaseModel):
    "Generic nonlinear 2-D (phase plane) oscillator."
    state = 'W V'
    input = 'c_0'
    param = 'a'
    const = {'tau': 1.0, 'I': 0.0, 'a': -2.0, 'b': -10.0, 'c': 0.0, 'd': 0.02,
             'e': 3.0, 'f': 1.0, 'g': 0.0, 'alpha': 1.0, 'beta': 1.0,
             'gamma': 1.0}
    drift = (
        'd * tau * (alpha*W - f*V**3 + e*V**2 + g*V + gamma*I + gamma*c_0)',
        'd * (a + b*V + c*V**2 - beta*W) / tau'
    )
    diffs = 1e-3, 1e-3
    obsrv = 'W', 'V'