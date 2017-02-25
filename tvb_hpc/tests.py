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
import pymbolic as pm
import numpy as np
import numpy.testing
from scipy.stats import kstest
from .bold import BalloonWindkessel
from .compiler import Compiler, CppCompiler, CompiledKernel, Spec
from .coupling import (Linear as LCf, Diff, Sigmoidal, Kuramoto as KCf,
    PostSumStat, BaseCoupling)
from .model import BaseModel, _TestModel, HMJE, RWW, JansenRit, Linear, G2DO
from .model import Kuramoto
from .network import DenseNetwork, DelayNetwork
from .rng import RNG
from .scheme import euler_maruyama_logp, EulerStep, EulerMaryuyamaStep
# from .harness import SimpleTimeStep
from .utils import getLogger, VarSubst


LOG = logging.getLogger(__name__)


class TestCase(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.tic = time.time()
        self.logger = getLogger(self.id())
        self._time_limit = 0.1

    def timeit(self, fn, *args, **kwds):
        niter = 0
        tic = toc = time.time()
        while ((toc - tic) < self._time_limit):
            fn(*args, **kwds)
            toc = time.time()
            niter += 1
        per_iter = self._time_limit / niter
        self.logger.info('%r requires %.3f ms / iter',
                         fn, per_iter * 1e3)

    def tearDown(self):
        super().tearDown()
        msg = 'required %.3fs'
        self.logger.info(msg, time.time() - self.tic)


class TestUtils(TestCase):

    def test_var_subst(self):
        subst = VarSubst(b=pm.parse('b[i, j]'))
        expr = subst(pm.parse('a + b * pre_syn[i, j]'))
        self.assertEqual(str(expr), 'a + b[i, j]*pre_syn[i, j]')


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
            LOG.debug('%s -> %s', var, expr)


class TestModel(TestCase):

    def setUp(self):
        super().setUp()
        self.spec = Spec('float', 8)

    def _test(self, model: BaseModel, spec: Spec, log_code=False):
        knl = model.kernel(target=CTarget())
        cknl = CompiledKernel(knl)
        arrs = model.prep_arrays(2048, self.spec)
        nblock, _, width = arrs[0].shape
        self.timeit(self._to_time, cknl, nblock, width, arrs)

    def _to_time(self, cknl, nblock, width, arrs):
        cknl(nblock, width, *arrs)

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
        self.spec = Spec('float', 8)

    def test_linear(self):
        model = G2DO()
        cf: BaseCoupling = LCf(model)
        self.assertEqual(cf.post_stat(0), PostSumStat.sum)

    def test_diff(self):
        model = G2DO()
        cf = Diff(model)
        self.assertEqual(cf.post_stat(0), PostSumStat.sum)

    def test_sigm(self):
        model = JansenRit()
        cf = Sigmoidal(model)
        self.assertEqual(cf.post_stat(0), PostSumStat.sum)

    def test_kura(self):
        model = Kuramoto()
        cf = KCf(model)
        self.assertEqual(cf.post_stat(0), PostSumStat.mean)


class TestNetwork(TestCase):

    def _test_dense(self, Model, Cfun):
        model = Model()
        cfun = Cfun(model)
        net = DenseNetwork(model, cfun)
        knl = net.kernel(target=CTarget())
        CompiledKernel(knl)

        net = DelayNetwork(model, cfun)
        knl = net.kernel(target=CTarget())
        CompiledKernel(knl)

    def test_hmje(self):
        self._test_dense(HMJE, LCf)

    def test_kuramoto(self):
        self._test_dense(Kuramoto, KCf)

    def test_jr(self):
        self._test_dense(JansenRit, Sigmoidal)


class TestScheme(TestCase):

    def test_euler_dt_literal(self):
        scheme = EulerStep(0.1)
        knl = scheme.kernel(target=CTarget())
        CompiledKernel(knl)

    def test_euler_dt_var(self):
        scheme = EulerStep(pm.var('dt'))
        knl = scheme.kernel(target=CTarget())
        CompiledKernel(knl)

    def test_em_dt_literal(self):
        scheme = EulerMaryuyamaStep(0.1)
        knl = scheme.kernel(target=CTarget())
        CompiledKernel(knl)

    def test_em_dt_var(self):
        scheme = EulerMaryuyamaStep(pm.var('dt'))
        knl = scheme.kernel(target=CTarget())
        CompiledKernel(knl)
