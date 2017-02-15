
from unittest import TestCase
import ctypes as ct

from tvb_hpc.compiler import Compiler
from tvb_hpc.model import test_model
from tvb_hpc.bold import BalloonWindkessel
from tvb_hpc.schemes import euler_maruyama_logp
from tvb_hpc.codegen import generate_code


class TestLogProb(TestCase):

    def setUp(self):
        self.model = test_model()

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
        lib = comp(generate_code(model, spec))
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
        model = test_model()
        fn = self._build_func(model, self.spec)
        arrs = model.prep_arrays(1024, self.spec)
        self._call(fn, *arrs)
        # TODO timing
        # TODO allclose against TVB

    def test_balloon_model(self):
        model = BalloonWindkessel()
        fn = self._build_func(model, self.spec)
