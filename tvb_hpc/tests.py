import logging
from unittest import TestCase
import ctypes as ct

from tvb_hpc.compiler import Compiler
from tvb_hpc.model import _TestModel, HMJE, RWW, JansenRit, Linear, G2DO
from tvb_hpc.bold import BalloonWindkessel
from tvb_hpc.schemes import euler_maruyama_logp
from tvb_hpc.codegen import generate_model_code


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
        self.spec = {
            'float': 'float',
            'width': 8,
            'align': 64,
            'cc': '/usr/local/bin/gcc-6',
            'cflags': '-mavx2 -O3 -ffast-math -fopenmp'.split()
        }

    def _build_func(self, model, spec):
        comp = Compiler(cc=spec['cc'], cflags=spec['cflags'])
        lib = comp(generate_model_code(model, spec))
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

    def test_pyopencl(self):

        import numpy as np
        import pyopencl as cl
        import pyopencl.clrandom as clr
        import pyopencl.array as cla

        context = cl.create_some_context()
        queue = cl.CommandQueue(context)

        array = cla.Array(queue, (1024, 1024), np.float32)

        rng = clr.PhiloxGenerator(context)

        rng.fill_normal(array)
        data = array.get(queue)

        self.assertTrue(data.mean()**2 < 1e-4)
        self.assertTrue((data.std() - 1)**2 < 1e-4)

    def test_r123_unifrom(self):
        import numpy as np
        from tvb_hpc.rng import RNG
        from scipy.stats import kstest
        import time
        rng = RNG()
        rng.comp.cflags += '-O3 -fopenmp'.split()
        rng.build()
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
