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

import time
import logging
import unittest

import numpy as np
import numpy.testing
from scipy.stats import kstest

from .bold import BalloonWindkessel
from .codegen.base import BaseSpec, Loop, Func, TypeTable
from .codegen.model import ModelGen1
from .codegen.cfun import CfunGen1
from .codegen.network import NetGen1, NetGen2
from .codegen.scheme import EulerSchemeGen
from .compiler import Compiler
from .compiler import CppCompiler
from .coupling import BaseCoupling
from .coupling import Linear as LCf, Diff, Sigmoidal, Kuramoto as KCf
from .model import BaseModel, _TestModel, HMJE, RWW, JansenRit, Linear, G2DO
from .model import Kuramoto
from .network import DenseNetwork, DenseDelayNetwork
from .rng import RNG
from .schemes import euler_maruyama_logp
from .harness import SimpleTimeStep
from .utils import getLogger

LOG = logging.getLogger(__name__)


class TestCase(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.tic = time.time()
        self.logger = getLogger(self.id())

    def tearDown(self):
        super().tearDown()
        msg = 'required %.3fs'
        self.logger.info(msg, time.time() - self.tic)


class TestCG(TestCase):

    def test_func_loop(self):
        ttable = TypeTable()
        for name in 'float double int'.split():
            ft = ttable.find_type(name)
            func = Func(
                name='test_loop',
                args=ft.name + ' *x',
                body=Loop(var='i', hi=10, body='x[i] += 1;')
            )
            x = np.zeros((20, ), ft.numpy)
            func(x)
            self.assertTrue((x[:10] == 1).all())
            self.assertTrue((x[10:] == 0).all())


class TestLogProb(TestCase):

    def setUp(self):
        super().setUp()
        self.model = _TestModel()

    def test_partials(self):
        logp = euler_maruyama_logp(
            self.model.state_sym,
            self.model.drift_sym,
            self.model.diffs_sym).sum()
        for var, expr in zip(self.model.indvars,
                             self.model.partial(logp)):
            pass


class TestModel(TestCase):

    def setUp(self):
        super().setUp()
        self.spec = BaseSpec('float', 8)

    def _test(self, model: BaseModel, spec: BaseSpec, log_code=False):
        comp = Compiler()
        cg = ModelGen1(model)
        code = cg.generate_code(spec)
        if log_code:
            LOG.debug(code)
        lib = comp(model.__class__.__name__, code)
        cg.func.fn = getattr(lib, cg.kernel_name)
        cg.func.annot_types(cg.func.fn)
        arrs = model.prep_arrays(1024, self.spec)
        cg.func(arrs[0].shape[0], *arrs)

    def test_balloon_model(self):
        model = BalloonWindkessel()
        self._test(model, self.spec)

    def test_hmje(self):
        model = HMJE()
        self._test(model, self.spec)

    def test_rww(self):
        model = RWW()
        self._test(model, self.spec)

    def test_jr(self):
        model = JansenRit()
        self._test(model, self.spec)

    def test_linear(self):
        model = Linear()
        self._test(model, self.spec)

    def test_g2do(self):
        model = G2DO()
        self._test(model, self.spec)


class TestRNG(TestCase):

    def test_r123_unifrom(self):
        comp = CppCompiler()
        rng = RNG(comp)
        rng.build(BaseSpec())
        # LOG.debug(list(comp.cache.values())[0]['asm'])
        array = np.zeros((1024 * 1024, ), np.float32)
        rng.fill(array)
        d, p = kstest(array, 'norm')
        # check normal samples are normal
        self.assertAlmostEqual(array.mean(), 0, places=2)
        self.assertAlmostEqual(array.std(), 1, places=2)
        self.assertLess(d, 0.01)


class TestCoupling(TestCase):

    def setUp(self):
        super().setUp()
        self.spec = BaseSpec('float', 8)

    def _test_cfun_code(self, cf: BaseCoupling):
        comp = Compiler()
        cg = CfunGen1(cf)
        dll = comp(cf.__class__.__name__, cg.generate_code(self.spec))
        for name in cg.kernel_names:
            getattr(dll, name)

    def test_linear(self):
        model = G2DO()
        self._test_cfun_code(LCf(model))

    def test_diff(self):
        model = G2DO()
        self._test_cfun_code(Diff(model))

    def test_sigm(self):
        model = JansenRit()
        self._test_cfun_code(Sigmoidal(model))

    def test_kura(self):
        model = Kuramoto()
        self._test_cfun_code(KCf(model))


class TestNetwork(TestCase):

    def setUp(self):
        super().setUp()
        self.spec = BaseSpec('float', 8)
        self.comp = Compiler()

    def _test_dense(self, model_cls, cfun_cls):
        model = model_cls()
        cfun = cfun_cls(model)
        net = DenseNetwork(model, cfun)
        cg = NetGen1(net)
        code = cg.generate_code(CfunGen1(cfun), self.spec)
        dll = self.comp('dense_net', code)
        cg.func.fn = getattr(dll, cg.kernel_name)
        cg.func.annot_types(cg.func.fn)
        nnode = 128
        nblck = int(nnode / self.spec.width)
        _, input, _, _, _, obsrv = model.prep_arrays(nblck, self.spec)
        robsrv = np.random.randn(*obsrv.shape).astype(self.spec.np_dtype)
        weights = np.random.randn(nnode, nnode).astype(self.spec.np_dtype)
        obsrv[:] = robsrv
        cg.func(nnode, weights, input, obsrv)
        input1 = input.copy()
        _, input, _, _, _, obsrv = model.prep_arrays(nblck, self.spec)
        obsrv[:] = robsrv
        net.npeval(weights, obsrv, input)
        input2 = input.copy()
        numpy.testing.assert_allclose(input1, input2, 1e-5, 1e-6)

    def test_hmje(self):
        self._test_dense(HMJE, LCf)

    def test_kuramoto(self):
        self._test_dense(Kuramoto, KCf)

    def test_jr(self):
        self._test_dense(JansenRit, Sigmoidal)

    def test_delay(self):
        model = HMJE()
        cfun = LCf(model)
        net = DenseDelayNetwork(model, cfun)
        cg = NetGen2(net)
        code = cg.generate_code(CfunGen1(cfun), self.spec)
        dll = self.comp('dense_delay_net', code)
        cg.func.fn = getattr(dll, cg.kernel_name)
        cg.func.annot_types(cg.func.fn)


class TestScheme(TestModel):

    def setUp(self):
        super().setUp()
        self.spec = BaseSpec('float', 8, openmp=True)
        self.comp = Compiler(openmp=True)

    def _test(self, model: BaseModel, spec: BaseSpec):
        comp = Compiler()
        modelcg = ModelGen1(model)
        eulercg = EulerSchemeGen(modelcg)
        code = eulercg.generate_c(self.spec)
        lib = comp(eulercg.kernel_name, code)
        eulercg.func.fn = getattr(lib, eulercg.kernel_name)
        eulercg.func.annot_types(eulercg.func.fn)
        arrs = model.prep_arrays(256, self.spec)
        xr = np.random.randn(*arrs[0].shape).astype('f')
        arrs[0][:] = xr
        nnode = arrs[0].shape[0] * arrs[0].shape[-1]
        eulercg.func(model.dt, nnode, *arrs)
        x0, _, _, _, _, o0 = arrs
        arrs = model.prep_arrays(256, self.spec)
        arrs[0][:] = xr
        model.npeval(arrs)
        x, i, p, f, g, o1 = arrs
        x1 = x + model.dt * f
        numpy.testing.assert_allclose(x0, x1, 1e-5, 1e-6)
        numpy.testing.assert_allclose(o0, o1, 1e-5, 1e-6)


class TestHarness(TestCase):

    def test_hmje(self):
        model = HMJE()
        cfun = LCf(model)
        net = DenseNetwork(model, cfun)
        stepper = SimpleTimeStep(model, cfun, net)
        nnode = 64
        dtype = stepper.spec.np_dtype
        weights = np.random.randn(nnode, nnode).astype(dtype)
        stepper.prep_data(weights)
        xr = np.random.randn(*stepper.arrs[0].shape).astype(dtype)
        stepper.arrs[0][:] = xr
        x, i, p, f, g, o = [a.copy() for a in stepper.arrs]
        niter = 100
        stepper.run(niter)
        obs1 = stepper.obsrv
        for _ in range(niter):
            net.npeval(weights, o, i)
            model.npeval((x, i, p, f, g, o))
            x += model.dt * f
        obs2 = o.copy()
        numpy.testing.assert_allclose(obs1, obs2, 1e-5, 1e-6)


if __name__ == '__main__':
    # list cases or methods
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd in ('cases', 'methods'):
            for key in dir():
                val = globals().get(key)
                if isinstance(val, type) and issubclass(val, TestCase):
                    if cmd == 'cases':
                        print('tvb_hpc.tests.%s' % (val.__name__, ))
                    elif cmd == 'methods':
                        for key in dir(val):
                            if key.startswith('test_'):
                                print('tvb_hpc.tests.%s.%s' % (
                                    val.__name__, key))
