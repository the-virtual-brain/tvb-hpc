#     Copyright 2017 TVB-HPC contributors
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.


"""
The model module describes neural mass models.

"""

import numpy as np
import pymbolic as pm
import loopy as lp
from pymbolic.mapper.differentiator import DifferentiationMapper
from pymbolic.mapper.c_code import CCodeMapper
from .compiler import Spec
from .utils import simplify, vars, exprs


class BaseModel:
    """
    BaseModel parses attributes on subclasses, defining a networked SDE.

    """

    dt = 1e-3
    state = ''
    input = ''
    param = ''
    drift = '',
    diffs = '',
    obsrv = '',
    const = {}
    auxex = []
    limit = []

    template = "#include <math.h>\n{func_code}"

    def __init__(self):
        # TODO check dependencies etc
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

    def prep_arrays(self, nnode, spec: Spec):
        """
        Prepare arrays for use with this model.

        """
        dtype = spec.np_dtype
        arrs = []
        for key in 'state input param drift diffs obsrv'.split():
            shape = nnode, len(getattr(self, key + '_sym')), spec.width
            arrs.append(np.zeros(shape, dtype))
        state = arrs[0]
        for i, (lo, hi) in enumerate(self.limit):
            state[:, i, :] = np.random.uniform(float(lo), float(hi),
                                               size=(nnode, spec.width))
        param = arrs[2]
        for i, psym in enumerate(self.param_sym):
            param[:, i, :] = self.const[psym.name]
        return arrs

    def kernel(self, target=None, typed=True):
        domains = self._kernel_domains()
        body = '\n'.join(self.instructions())
        data = self._kernel_data()
        knl = lp.make_kernel(domains, body, data, target=target)
        if typed:
            dtypes = self._kernel_dtypes()
            knl = lp.add_dtypes(knl, dtypes)
        return knl

    def _kernel_domains(self):
        return "{ [i, j]: 0 <= i < nblock and 0 <= j < width }"

    def _kernel_data(self):
        data = 'nblock width state input param drift diffs obsrv'.split()
        return data

    def _kernel_dtypes(self):
        dtypes = {key: np.float32 for key in self._kernel_data()[2:]}
        dtypes['nblock'] = np.uintc
        dtypes['width'] = np.uintc
        return dtypes

    def _insn_constants(self):
        fmt = '<> {key} = {val}'
        for key, val in self.const.items():
            if key not in self.param:
                yield fmt.format(key=key, val=val)

    def _insn_unpack(self):
        fmt = '<> {var} = {kind}[i, {i}, j]'
        for kind in 'state input param'.split():
            vars = getattr(self, kind + '_sym')
            for i, var in enumerate(vars):
                yield fmt.format(kind=kind, var=var.name, i=i)

    def _insn_auxex(self):
        fmt = '<> {lhs} = {rhs}'
        for lhs, rhs in self.auxex:
            yield fmt.format(lhs=lhs, rhs=rhs)

    def _insn_store(self):
        fmt = '{kind}[i, {i}, j] = {expr}'
        for kind in 'drift diffs obsrv'.split():
            exprs = getattr(self, kind + '_sym')
            nvar = len(getattr(self, kind + '_sym'))
            for i, expr in enumerate(exprs):
                yield fmt.format(kind=kind, expr=str(expr), i=i)

    def instructions(self):
        lines = []
        for section in 'constants unpack auxex store'.split():
            lines += list(getattr(self, '_insn_' + section)())
        return lines

class _TestModel(BaseModel):
    state = 'y1 y2'
    input = 'i1'
    param = 'a b c'
    auxex = [('y1_3', 'y1 * y1 * y1')]
    drift = '(y1 - y1_3/3 + y2 + e)*b', '(a - y1 + i1 + d)/b'
    diffs = 'c', 'c'
    obsrv = 'y1',
    const = {'d': 3.0, 'e': -12.23904e-2, 'a': 1.05, 'b': 3.0, 'c': 1e-5}
    limit = (-1, 1), (-1, 1)


class Kuramoto(BaseModel):
    "Kuramoto model of phase synchronization."
    state = 'theta'
    limit = (-np.pi, np.pi),
    input = 'I'
    param = 'omega'
    drift = 'omega + I',
    diffs = 0,
    obsrv = 'theta', 'sin(theta)'
    const = {'omega': 1.0}


class HMJE(BaseModel):
    "Hindmarsh-Rose-Jirsa Epileptor model of seizure dynamics."
    state = 'x1 y1 z x2 y2 g'
    limit = (-2, 1), (20, 2), (2, 5), (-2, 0), (0, 2), (-1, 1)
    input = 'c1 c2'
    param = 'x0 Iext r'
    drift = (
    'tt * (y1 - z + Iext + Kvf * c1 + ('
            '  (x1 <  0)*(-a * x1 * x1 + b * x1)'           # noqa: E131
            '+ (x1 >= 0)*(slope - x2 + 0.6 * (z - 4)**2)'   # noqa: E131
        ') * x1)',
    'tt * (c - d * x1 * x1 - y1)',
    'tt * (r * (4 * (x1 - x0) - z +  Ks * c1))',
    'tt * (-y2 + x2 - x2*x2*x2 + Iext2 + 2 * g - 0.3 * (z - 3.5) + Kf * c2)',
    'tt * ((-y2 + (x2 >= (-0.25)) * (aa * (x2 + 0.25))) / tau)',
    'tt * (-0.01 * (g - 0.1 * x1))'
    )
    diffs = 0, 0, 0, 0.0003, 0.0003, 0
    obsrv = 'x1', 'x2', 'z', '-x1 + x2'
    const = {'Iext2': 0.45, 'a': 1.0, 'b': 3.0, 'slope': 0.0, 'tt': 1.0, 'c': 1.0,
             'd': 5.0, 'Kvf': 0.0, 'Ks': 0.0, 'Kf': 0.0, 'aa': 6.0, 'tau': 10.0,
             'x0': -1.6, 'Iext': 3.1, 'r': 0.00035}


class RWW(BaseModel):
    "Reduced Wong-Wang firing rate model."
    state = 'S'
    limit = (0, 1),
    input = 'c'
    param = 'w io'
    const = {'a': 0.270, 'b': 0.108, 'd': 154.0, 'g': 0.641,
             'ts': 100.0, 'J': 0.2609, 'w': 0.6, 'io': 0.33}
    auxex = [
        ('x', 'w * J * S + io + J * c'),
        ('h', '(a * x - b) / (1 - exp(-d*(a*x - b)))')
    ]
    drift = '- (S / ts) + (1 - S) * h * g',
    diffs = 0.01,
    obsrv = 'S',


class JansenRit(BaseModel):
    "Jansen-Rit model of visual evoked potentials."
    state = 'y0 y1 y2 y3 y4 y5'
    limit = (-1, 1), (-500, 500), (-50, 50), (-6, 6), (-20, 20), (-500, 500)
    const = {'A': 3.25, 'B': 22.0, 'a': 0.1, 'b': 0.05, 'v0': 5.52,
             'nu_max': 0.0025, 'r': 0.56, 'J': 135.0, 'a_1': 1.0, 'a_2': 0.8,
             'a_3': 0.25, 'a_4': 0.25, 'mu': 0.22}
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
    limit = (-10, 10),
    input = 'c'
    param = 'lam'
    const = {'lam': -1}  # default value
    drift = 'lam * x + c',
    diffs = 1e-2,
    obsrv = 'x',


class G2DO(BaseModel):
    "Generic nonlinear 2-D (phase plane) oscillator."
    state = 'W V'
    limit = (-5, 5), (-5, 5)
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
