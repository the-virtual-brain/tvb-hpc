#     Copyright 2018 TVB-HPC contributors
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
from scipy.stats import kstest
from .bold import BalloonWindkessel
from .coupling import (
    Linear as LCf, Diff, Sigmoidal, Kuramoto as KCf,
    PostSumStat)
from .model import BaseModel, _TestModel, HMJE, RWW, JansenRit, Linear, G2DO
from .model import Kuramoto
from .network import Network
from .rng import RNG
from .scheme import euler_maruyama_logp, EulerStep, EulerMaryuyamaStep
# from .harness import SimpleTimeStep
from .numba import NumbaTarget
from .utils import getLogger, VarSubst
from .workspace import Workspace, CLWorkspace


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


class BaseTestCl(TestCase):

    def setUp(self):
        try:
            import pyopencl as cl
            self.ctx = cl.create_some_context(interactive=False)
            self.cq = cl.CommandQueue(self.ctx)
        except Exception as exc:
            raise unittest.SkipTest(
                'unable to create CL queue (%r)' % (exc, ))
        self.target = lp.target.pyopencl.PyOpenCLTarget()
        super().setUp()


class TestCl(BaseTestCl):

    def test_copy(self):
        knl = lp.make_kernel('{:}', 'a = b')
        knl = lp.to_batched(knl, 16, 'a b'.split(), 'i', sequential=False)
        knl = lp.add_and_infer_dtypes(knl, {'a,b': 'f'})
        import pyopencl.array as ca
        a = ca.zeros(self.cq, (16, ), 'f')
        b = ca.zeros(self.cq, (16, ), 'f')
        b[:] = np.r_[:16].astype('f')
        knl(self.cq, a=a, b=b)
        np.testing.assert_allclose(a.get(), b.get())

    def test_add_loops(self):
        # build kernel
        kernel = """
        <> dx = a * x + b * y
        <> dy = c * x + d * y
        xn = x + dt * dx {nosync=*}
        yn = y + dt * dy {nosync=*}
        """
        state = 'x y xn yn'.split()
        knl = lp.make_kernel("{:}", kernel)
        knl = lp.add_and_infer_dtypes(knl, {'a,b,c,d,x,y,dt,xn,yn': 'f'})
        knl = lp.to_batched(knl, 'nt', state, 'it')
        knl = lp.to_batched(knl, 'na', state + ['a'], 'ia')
        knl = lp.to_batched(knl, 'nb', state + ['b'], 'ib')
        knl = lp.tag_inames(knl, [('ia', 'g.0'), ('ib', 'l.0')], force=True)
        # setup pyopencl
        import pyopencl as cl
        import pyopencl.array as ca
        import numpy as np
        ctx = cl.create_some_context(interactive=False)
        cq = cl.CommandQueue(ctx)
        # workspace
        a = ca.Array(cq, (10,), 'f')
        b = ca.Array(cq, (10,), 'f')
        x = ca.Array(cq, (10, 10, 5), 'f')
        y = ca.Array(cq, (10, 10, 5), 'f')
        xn = ca.Array(cq, (10, 10, 5), 'f')
        yn = ca.Array(cq, (10, 10, 5), 'f')
        a[:], b[:] = np.random.rand(2, 10).astype('f')
        c, d, dt = [np.float32(_) for _ in (0.5, 0.6, 0.1)]
        x[:], y[:], xn[:], yn[:] = np.random.rand(4, 10, 10, 5).astype('f')
        # execute
        knl(cq,
            na=np.int32(a.size),
            nb=np.int32(b.size),
            nt=np.int32(x.shape[-1]),
            a=a, b=b, c=c, d=d, x=x, y=y, dt=dt, xn=xn, yn=yn)
        # cl arr doesn't broadcast
        a_ = ca.Array(cq, (10, 10, 5), 'f')
        b_ = ca.Array(cq, (10, 10, 5), 'f')
        a_[:] = np.tile(a.get()[:, None], (10, 1, 5)).astype('f')
        b_[:] = np.tile(b.get()[:, None, None], (1, 10, 5)).astype('f')
        # check
        np.testing.assert_allclose(
            xn.get(), (x + dt * (a_ * x + b_ * y)).get(), 1e-6, 1e-6)
        np.testing.assert_allclose(
            yn.get(), (y + dt * (c * x + d * y)).get(), 1e-6, 1e-6)


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

    def _dtype_and_code(self, knl, **extra_dtypes):
        dtypes = {'in': np.float32, 'out': np.float32}
        dtypes.update(extra_dtypes)
        knl = lp.add_dtypes(knl, dtypes)
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

    def test_wrap_loop_with_param(self):
        knl = lp.make_kernel("{[i,j]:0<=i,j<n}",
                             """
                             <> a = a_values[i]
                             out[i] = a * sum(j, (i/j)*in[i, j])
                             """,
                             target=CTarget())
        # in will depend on t
        knl2 = lp.to_batched(knl, 'T', ['in'], 't', sequential=True)
        print(self._dtype_and_code(knl2, a_values=np.float32))

    def test_split_iname3(self):
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
        target = NumbaTarget()
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
            target=target)
        knl = lp.add_and_infer_dtypes(knl, {
            'out,dat,vec': np.float32,
            'col,row,n,nnz': np.uintc,
        })
        # col and dat have uninferrable shape
        knl.args[3].shape = pm.var('nnz'),
        knl.args[4].shape = pm.var('nnz'),
        from scipy.sparse import csr_matrix
        n = 64
        mat = csr_matrix(np.ones((64, 64)) * (np.random.rand(64, 64) < 0.1))
        row = mat.indptr.astype(np.uintc)
        col = mat.indices.astype(np.uintc)
        dat = mat.data.astype(np.float32)
        out, vec = np.random.rand(2, n).astype(np.float32)
        nnz = mat.nnz
        knl(n, nnz, row, col, dat, vec, out)
        np.testing.assert_allclose(out, mat * vec, 1e-5, 1e-6)


class TestNumbaTarget(TestCase):

    def test_simple(self):
        target = NumbaTarget()
        knl = lp.make_kernel(
            "{ [i]: 0<=i<n }",
            "out[i] = 2*a[i]",
            target=target
        )
        typed = lp.add_dtypes(knl, {'a': np.float32})
        a, out = np.zeros((2, 10), np.float32)
        a[:] = np.r_[:a.size]
        typed(a, 10, out)
        np.testing.assert_allclose(out, a * 2)


class TestCompiledKernel(TestCase):

    @unittest.skip
    def test_simple_kernel(self):
        knl = lp.make_kernel(
            "{ [i]: 0<=i<n }",
            "out[i] = 2*a[i]",
            target=CTarget()
        )
        typed = lp.add_dtypes(knl, {'a': np.float32})
        code, _ = lp.generate_code(typed)
        fn = CompiledKernel(typed)  # noqa
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

    def _test(self, model: BaseModel, log_code=False):
        target = NumbaTarget()
        knl = model.kernel(target=target)
        target.get_kernel_executor(knl)

    def test_balloon_model(self):
        model = BalloonWindkessel()
        self._test(model)

    def test_hmje(self):
        model = HMJE()
        self._test(model)

    def test_rww(self):
        model = RWW()
        self._test(model)

    def test_jr(self):
        model = JansenRit()
        self._test(model)

    def test_linear(self):
        model = Linear()
        self._test(model)

    def test_g2do(self):
        model = G2DO()
        self._test(model)


class TestRNG(TestCase):

    # Trickier to use Numba. Can we port one of them?
    @unittest.skip
    def test_r123_normal(self):
        rng = RNG()
        rng.build()
        array = np.zeros((1024 * 1024, ), np.float32)
        rng.fill(array)
        d, p = kstest(array, 'norm')
        # check normal samples are normal
        self.assertAlmostEqual(array.mean(), 0, places=2)
        self.assertAlmostEqual(array.std(), 1, places=2)
        self.assertLess(d, 0.01)


class TestCoupling(TestCase):

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
        net = Network(model, cfun)
        target = NumbaTarget()
        knl = net.kernel(target=target)
        target.get_kernel_executor(knl)

    def test_hmje(self):
        self._test_dense(HMJE, LCf)

    def test_kuramoto(self):
        self._test_dense(Kuramoto, KCf)

    def test_jr(self):
        self._test_dense(JansenRit, Sigmoidal)


class TestScheme(TestCase):

    def _test_scheme(self, scheme):
        target = NumbaTarget()
        knl = scheme.kernel(target=target)
        target.get_kernel_executor(knl)

    def test_euler_dt_literal(self):
        self._test_scheme(EulerStep(0.1))

    def test_euler_dt_var(self):
        self._test_scheme(EulerStep(pm.var('dt')))

    def test_em_dt_literal(self):
        self._test_scheme(EulerMaryuyamaStep(0.1))

    def test_em_dt_var(self):
        self._test_scheme(EulerMaryuyamaStep(pm.var('dt')))


class TestHackathon(TestCase):
    pass


class WorkspaceTestsMixIn:
    def test_copy(self):
        knl = lp.make_kernel('{:}', 'a = b + c + x', target=self.target)
        knl = lp.to_batched(knl, 'm', ['a', 'b'], 'i')
        knl = lp.to_batched(knl, 'n', ['a', 'c'], 'j')
        knl = lp.add_and_infer_dtypes(knl, {'a,b,c,x': 'f'})
        wspc = self.make_workspace(knl, m=10, n=5, x=3.5)
        self.assertEqual(wspc.data['a'].shape, (5, 10))
        self.assertEqual(wspc.data['b'].shape, (10, ))
        self.assertEqual(wspc.data['x'].shape, ())
        self.assertEqual(wspc.data['x'].dtype, np.float32)


class TestWorkspaceNumba(TestCase, WorkspaceTestsMixIn):
    target = NumbaTarget()

    def make_workspace(self, *args, **kwargs):
        return Workspace(*args, **kwargs)


class TestWorkspaceCL(BaseTestCl, WorkspaceTestsMixIn):
    def make_workspace(self, *args, **kwargs):
        return CLWorkspace(self.cq, *args, **kwargs)


class TestMetrics(TestCase):

    def test_ocov(self):
        from dsl_cuda.tvb_hpc import OnlineCov
        ocov = OnlineCov()
        knl = ocov.kernel(NumbaTarget())
        _ = lp.generate_code(knl)
        self.assertTrue(_)

    def test_bcov(self):
        from dsl_cuda.tvb_hpc import BatchCov
        bcov = BatchCov()
        knl = bcov.kernel(NumbaTarget())
        self.assertTrue(lp.generate_code(knl))


class TestRmap(TestCase):

    def test_rmap_to_avg(self):
        from dsl_cuda.tvb_hpc import RMapToAvg
        knl = RMapToAvg().kernel(NumbaTarget())
        i = np.r_[:16].reshape((-1, 1))
        rmap = i // 4
        node = i.astype('f')
        roi = np.zeros((4, 1), 'f')
        knl(nroi=4, nvar=1, nnode=16, rmap=rmap, node=node, roi=roi)
        np.testing.assert_allclose(roi[:, 0], node.reshape((4, 4)).sum(axis=1))

    def test_rmap_from_avg(self):
        from dsl_cuda.tvb_hpc import RMapFromAvg
        knl = RMapFromAvg().kernel(NumbaTarget())
        i = np.r_[:16].reshape((-1, 1))
        rmap = i // 4
        node = np.zeros((16, 1), 'f')
        roi = np.r_[:4].reshape((4, 1)).astype('f')
        knl(nroi=4, nvar=1, nnode=16, rmap=rmap, node=node, roi=roi)
        np.testing.assert_allclose(rmap, node)
