import logging
from unittest import TestCase
import ctypes as ct

from tvb_hpc.codegen import BaseSpec
from tvb_hpc.compiler import Compiler
from tvb_hpc.model import (BaseModel, _TestModel, HMJE, RWW, JansenRit,
                           Linear, G2DO)
from tvb_hpc.bold import BalloonWindkessel
from tvb_hpc.schemes import euler_maruyama_logp


LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class TestLogProb(TestCase):

    def setUp(self):
        self.model = _TestModel()

    def test_partials(self):
        logp = euler_maruyama_logp(
            self.model.state_sym,
            self.model.drift_sym,
            self.model.diffs_sym).sum()
        for var, expr in zip(self.model.indvars,
                             self.model.partial(logp)):
            pass


class TestCodeGen(TestCase):

    def setUp(self):
        self.cflags = '-mavx2 -O3 -ffast-math -fopenmp'.split()
        self.spec = BaseSpec('float', 8, 64).dict

    def _build_func(self, model: BaseModel, spec):
        comp = Compiler()
        lib = comp(model.generate_code(spec))
        fn = getattr(lib, model.kernel_name)
        fn.restype = None  # equiv. C void return type
        ui = ct.c_uint
        f = {'float': ct.c_float, 'double': ct.c_double}[spec['float']]
        fp = ct.POINTER(f)
        fn.argtypes = [ui, fp, fp, fp, fp, fp, fp]
        return fn

    def _call(self, fn, *args):
        x, i, p, f, g, o = args
        nn = ct.c_uint(x.shape[0])
        fp = fn.argtypes[1]
        args = [a.ctypes.data_as(fp) for a in args]
        fn(nn, *args)

    def test_test_model_code_gen(self):
        model = _TestModel()
        fn = self._build_func(model, self.spec)
        arrs = model.prep_arrays(1024, self.spec)
        self._call(fn, *arrs)
        # TODO timing
        # TODO allclose against TVB

    def test_balloon_model(self):
        model = BalloonWindkessel()
        fn = self._build_func(model, self.spec)

    def test_hmje(self):
        model = HMJE()
        fn = self._build_func(model, self.spec)

    def test_rww(self):
        model = RWW()
        fn = self._build_func(model, self.spec)

    def test_jr(self):
        model = JansenRit()
        fn = self._build_func(model, self.spec)

    def test_linear(self):
        model = Linear()
        fn = self._build_func(model, self.spec)

    def test_g2do(self):
        model = G2DO()
        fn = self._build_func(model, self.spec)


class TestRNG(TestCase):

    def test_r123_unifrom(self):
        import numpy as np
        from tvb_hpc.rng import RNG
        from scipy.stats import kstest
        import time
        from tvb_hpc.compiler import CppCompiler
        comp = CppCompiler(gen_asm=True)
        comp.cflags += '-O3 -fopenmp'.split()
        rng = RNG(comp)
        rng.build()
        # LOG.debug(list(comp.cache.values())[0]['asm'])
        array = np.zeros((1024 * 1024, ), np.float32)
        rng.fill(array)
        d, p = kstest(array, 'norm')
        self.assertAlmostEqual(array.mean(), 0, places=2)
        self.assertAlmostEqual(array.std(), 1, places=2)
        self.assertLess(d, 0.01)

        # test timing
        shape = (1024 * 1024, )
        tic = time.time()
        array = np.zeros(shape)
        array = array.astype(np.float32)
        rng.fill(array)
        et1 = time.time() - tic

        tic = time.time()
        np.random.randn(*shape).astype('f')
        et2 = time.time() - tic
        self.assertLess(et1, et2)
        LOG.info('r123 %0.3fs, np %0.3fs' % (et1, et2))


class TestCoupling(TestCase):

    def setUp(self):
        # GCC 6 is generating AVX-512 instructions for functions declared SIMD
        # despite march=native... encouraging but not testable locally.
        self.cflags = '-O3 -march=native'.split()
        self.spec = BaseSpec('float', 8, 64)

    def _test_cfun_code(self, cf):
        comp = Compiler(cflags=self.cflags)
        dll = comp(cf.generate_code(self.spec))
        getattr(dll, cf._post_sum_name)
        getattr(dll, cf._pre_sum_name)

    def test_linear(self):
        from tvb_hpc.coupling import Linear
        from tvb_hpc.model import G2DO
        model = G2DO()
        self._test_cfun_code(Linear(model))

    def test_diff(self):
        from tvb_hpc.coupling import Diff
        from tvb_hpc.model import G2DO
        model = G2DO()
        self._test_cfun_code(Diff(model))

    def test_sigm(self):
        from tvb_hpc.coupling import Sigmoidal
        from tvb_hpc.model import JansenRit
        model = JansenRit()
        self._test_cfun_code(Sigmoidal(model))

    def test_kura(self):
        from tvb_hpc.coupling import Kuramoto as KCf
        from tvb_hpc.model import Kuramoto
        model = Kuramoto()
        self._test_cfun_code(KCf(model))


class TestNetwork(TestCase):

    def setUp(self):
        # GCC 6 is generating AVX-512 instructions for functions declared SIMD
        # despite march=native... encouraging but not testable locally.
        self.cflags = '-O3 -march=native'.split()
        self.spec = BaseSpec('float', 8, 64)
        self.comp = Compiler(cflags=self.cflags)

        from tvb_hpc.coupling import Sigmoidal
        from tvb_hpc.model import JansenRit
        self.model = JansenRit()
        self.cfun = Sigmoidal(self.model)


    def test_dense(self):
        from tvb_hpc.network import DenseNetwork
        net = DenseNetwork(self.model, self.cfun)
        code = net.generate_code(self.spec)
        print(code)
        dll = self.comp(code)
        getattr(dll, net.kernel_name)
