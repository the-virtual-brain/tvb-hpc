import ctypes as ct
import logging
from unittest import TestCase

import numpy as np
from scipy.stats import kstest

from .bold import BalloonWindkessel
from .codegen.base import BaseSpec, Loop, Func, TypeTable
from .codegen.model import ModelGen1
from .codegen.cfun import CfunGen1
from .codegen.network import NetGen1
from .compiler import Compiler
from .compiler import CppCompiler
from .coupling import BaseCoupling
from .coupling import Linear as LCf, Diff, Sigmoidal, Kuramoto as KCf
from .model import BaseModel, _TestModel, HMJE, RWW, JansenRit, Linear, G2DO
from .model import Kuramoto
from .network import DenseNetwork
from .rng import RNG
from .schemes import euler_maruyama_logp

LOG = logging.getLogger(__name__)


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
        self.spec = BaseSpec('float', 8)

    def _build_func(self, model: BaseModel, spec: BaseSpec, log_code=False):
        comp = Compiler()
        cg = ModelGen1(model)
        code = cg.generate_code(spec)
        if log_code:
            LOG.debug(code)
        lib = comp(model.__class__.__name__, code)
        fn = getattr(lib, cg.kernel_name)
        fn.restype = None  # equiv. C void return type
        ui = ct.c_uint
        f = spec.ct_dtype
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
        self._build_func(model, self.spec)

    def test_hmje(self):
        model = HMJE()
        self._build_func(model, self.spec)

    def test_rww(self):
        model = RWW()
        self._build_func(model, self.spec)

    def test_jr(self):
        model = JansenRit()
        self._build_func(model, self.spec)

    def test_linear(self):
        model = Linear()
        self._build_func(model, self.spec)

    def test_g2do(self):
        model = G2DO()
        self._build_func(model, self.spec)


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
        self.spec = BaseSpec('float', 8)
        self.comp = Compiler()

    def _test_dense(self, model_cls, cfun_cls):
        model = model_cls()
        cfun = cfun_cls(model)
        net = DenseNetwork(model, cfun)
        cg = NetGen1(net)
        code = cg.generate_code(CfunGen1(cfun), self.spec)
        dll = self.comp('dense_net', code)
        getattr(dll, cg.kernel_name)

    def test_hmje(self):
        self._test_dense(HMJE, LCf)

    def test_kuramoto(self):
        self._test_dense(Kuramoto, KCf)

    def test_jr(self):
        self._test_dense(JansenRit, Sigmoidal)
