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
import loopy as lp
from loopy.target.c import CTarget
import numpy as np
import numpy.testing
from scipy.stats import kstest
# from .bold import BalloonWindkessel
from .compiler import Compiler, CppCompiler, CompiledKernel, Spec
# from .coupling import Linear as LCf, Diff, Sigmoidal, Kuramoto as KCf
from .model import BaseModel, _TestModel, HMJE, RWW, JansenRit, Linear, G2DO
from .model import Kuramoto
# from .network import DenseNetwork, DenseDelayNetwork
from .rng import RNG
# from .scheme import euler_maruyama_logp, EulerSchemeGen
# from .harness import SimpleTimeStep
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


class TestCompiledKernel(TestCase):

    def test_simple_kernel(self):
        knl = lp.make_kernel("{ [i]: 0<=i<n }", "out[i] = 2*a[i]", target=CTarget())
        typed = lp.add_dtypes(knl, {'a': np.float32})
        code, _ = lp.generate_code(typed)
        fn = CompiledKernel(typed)
        a, out = np.zeros((2, 10), np.float32)
        a[:] = np.r_[:a.size]
        fn(a, 10, out)
        np.testing.assert_allclose(out, a * 2)


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
        self.spec = Spec('float', 8)

    def _test(self, model: BaseModel, spec: Spec, log_code=False):
        knl = model.kernel(target=CTarget())
        cknl = CompiledKernel(knl)
        arrs = model.prep_arrays(256, self.spec)
        nblock, _, width = arrs[0].shape
        cknl(nblock, width, *arrs)

    @unittest.skip('reimpl')
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

    def test_r123_normal(self):
        rng = RNG()
        rng.build(Spec())
        # LOG.debug(list(comp.cache.values())[0]['asm'])
        array = np.zeros((1024 * 1024, ), np.float32)
        rng.fill(array)
        d, p = kstest(array, 'norm')
        # check normal samples are normal
        self.assertAlmostEqual(array.mean(), 0, places=2)
        self.assertAlmostEqual(array.std(), 1, places=2)
        self.assertLess(d, 0.01)


@unittest.skip('reimpl')
class TestCoupling(TestCase):

    def setUp(self):
        super().setUp()
        self.spec = Spec('float', 8)

    def _test_cfun_code(self, Cf, model):
        cf = Cf(model, storage=Storage.default)
        comp = Compiler()
        # TODO improve API here:
        dll = comp(cf.__class__.__name__, cf.generate_code(self.spec))
        for name in cf.kernel_names:
            getattr(dll, name)

    def test_linear(self):
        model = G2DO()
        self._test_cfun_code(LCf, model)

    def test_diff(self):
        model = G2DO()
        self._test_cfun_code(Diff, model)

    def test_sigm(self):
        model = JansenRit()
        self._test_cfun_code(Sigmoidal, model)

    def test_kura(self):
        model = Kuramoto()
        self._test_cfun_code(KCf, model)


@unittest.skip('reimpl')
class TestNetwork(TestCase):

    def setUp(self):
        super().setUp()
        self.spec = Spec('float', 8)
        self.comp = Compiler()

    def _test_dense(self, model_cls, cfun_cls):
        model = model_cls()
        cfun = cfun_cls(model)
        net = DenseNetwork(model, cfun)
        code = net.generate_code(self.spec)
        dll = self.comp('dense_net', code)
        net.func.fn = getattr(dll, net.kernel_name)
        net.func.annot_types(net.func.fn)
        nnode = 128
        nblck = int(nnode / self.spec.width)
        _, input, _, _, _, obsrv = model.prep_arrays(nblck, self.spec)
        robsrv = np.random.randn(*obsrv.shape).astype(self.spec.np_dtype)
        weights = np.random.randn(nnode, nnode).astype(self.spec.np_dtype)
        obsrv[:] = robsrv
        net.func(nnode, weights, input, obsrv)
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
        spec = Spec('float', width=4)
        model = HMJE()
        cfun = LCf(model)
        net = DenseDelayNetwork(model, cfun)
        code = net.generate_code(spec)
        dll = self.comp('dense_delay_net', code)
        net.func.fn = getattr(dll, net.kernel_name)
        net.func.annot_types(net.func.fn)
        nnode = 64
        nblck = int(nnode / spec.width)
        _, input, _, _, _, obsrv = model.prep_arrays(nblck, spec)
        # robsrv = np.tile(np.random.randn(*obsrv.shape), (100, 1, 1, 1))
        obsrv = np.zeros((100, ) + obsrv.shape, obsrv.dtype)
        np.random.seed(42)
        robsrv = np.random.randn(*obsrv.shape).astype(spec.np_dtype)
        weights = np.abs(np.random.randn(nnode, nnode).astype(spec.np_dtype))
        delays = np.random.uniform(0, 50, size=(nnode, nnode)).astype(np.uintc)
        i_t = 75
        obsrv[:] = robsrv
        # import pdb; pdb.set_trace()
        net.func(nnode, i_t, delays, weights, input, obsrv)
        input1 = input.copy()
        _, input, _, _, _, obsrv = model.prep_arrays(nblck, spec)
        obsrv = np.zeros((100, ) + obsrv.shape, obsrv.dtype)
        obsrv[:] = robsrv
        net.npeval(i_t, delays, weights, obsrv, input, debug=spec.debug)
        input2 = input.copy()
        numpy.testing.assert_allclose(input1, input2, 1e-5, 1e-6)


@unittest.skip('reimpl')
class TestScheme(TestModel):

    def setUp(self):
        super().setUp()
        self.spec = Spec('float', 8, openmp=True)
        self.comp = Compiler(openmp=True)

    def _test(self, model: BaseModel, spec: Spec):
        comp = Compiler()
        eulercg = EulerSchemeGen(model)
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


@unittest.skip('reimpl')
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
