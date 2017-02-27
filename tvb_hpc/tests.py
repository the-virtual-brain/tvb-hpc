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
from .network import Network
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


class TestLoopTransforms(TestCase):
    """
    These are more tests to check that our use of Loopy is correct.

    """

    def setUp(self):
        super().setUp()
        from loopy.target.ispc import ISPCTarget
        target = ISPCTarget()
        self.knl = lp.make_kernel('{[i]:0<=i<n}', "out[i] = in[i]",
                                  target=target)

    def _dtype_and_code(self, knl):
        knl = lp.add_dtypes(knl, {'in': np.float32, 'out': np.float32})
        code, _ = lp.generate_code(knl)
        return code

    def test_chunk_iname(self):
        "Chunk useful to split work for e.g. omp par for"
        knl = lp.chunk_iname(self.knl, 'i', 8)
        print(self._dtype_and_code(knl))

    def test_split_iname2(self):
        "Split useful for omp simd inner loop"
        knl = lp.split_iname(self.knl, 'i', 8)
        knl = lp.tag_inames(knl, [('i_inner', 'ilp.unr',)])
        print(self._dtype_and_code(knl))

    def test_wrap_loop(self):
        "Take kernel, place in larger loop, offsetting certain vars"
        knl = lp.make_kernel("{[i,j]:0<=i,j<n}",
                             "out[i] = sum(j, (i/j)*in[i, j])",
                             target=CTarget())
        # in will depend on t
        knl2 = lp.to_batched(knl, 'T', ['in'], 't')
        print(self._dtype_and_code(knl2))

    #@unittest.skip('question asked on mailing list')
    def test_wrap_loop_with_param(self):
        knl = lp.make_kernel("{[i,j]:0<=i,j<n}",
                             """
                             <> a = a_values[i]
                             out[i] = a * sum(j, (i/j)*in[i, j])
                             """,
                             target=CTarget())
        # in will depend on t
        knl2 = lp.to_batched(knl, 'T', ['in'], 't', sequential=True)
        print(self._dtype_and_code(knl2))

    def test_split_iname2(self):
        "Split one of two inames."
        from loopy.target.ispc import ISPCTarget as CTarget
        knl = lp.make_kernel("{[i,j]:0<=i,j<n}",
                             "out[i, j] = in[i, j]",
                             target=CTarget())
        knl = lp.split_iname(knl, 'i', 8)
        knl = lp.prioritize_loops(knl, ['i_outer', 'j', 'i_inner'])
        print(self._dtype_and_code(knl))

    def test_sparse_matmul(self):
        "Tests how to do sparse indexing w/ loop."
        knl = lp.make_kernel(
            [
                '{[i]: 0   <= i <   n}',
                # note loop bounded by jlo jhi
                '{[j]: jlo <= j < jhi}'
            ],
            # which are set as instructions
            """
            <> jlo = row[i]
            <> jhi = row[i + 1]
            out[i] = sum(j, dat[j] * vec[col[j]])
            """,
            'n nnz row col dat vec out'.split(),
            target=CTarget())
        knl = lp.add_and_infer_dtypes(knl, {
            'out,dat,vec': np.float32,
            'col,row,n,nnz': np.uintc,
        })
        # col and dat have uninferrable shape
        knl.args[3].shape = pm.var('nnz'),
        knl.args[4].shape = pm.var('nnz'),
        cknl = CompiledKernel(knl)
        from scipy.sparse import csr_matrix
        n = 64
        mat = csr_matrix(np.ones((64, 64)) * (np.random.rand(64, 64) < 0.1))
        row = mat.indptr.astype(np.uintc)
        col = mat.indices.astype(np.uintc)
        dat = mat.data.astype(np.float32)
        out, vec = np.random.rand(2, n).astype(np.float32)
        nnz = mat.nnz
        cknl(n, nnz, row, col, dat, vec, out)
        np.testing.assert_allclose(out, mat * vec, 1e-5, 1e-6)




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
