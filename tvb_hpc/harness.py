
"""
Harnesses hold things in place.

"""

from tvb_hpc.utils import timer
from tvb_hpc.codegen.cfun import CfunGen1
from tvb_hpc.codegen.model import ModelGen1
from tvb_hpc.codegen.scheme import EulerSchemeGen
from tvb_hpc.codegen.network import NetGen1
from tvb_hpc.codegen.base import BaseSpec
from tvb_hpc.compiler import Compiler
from tvb_hpc.utils import getLogger


class BaseHarness:
    "Type tag."


class SimpleTimeStep(BaseHarness):
    """
    Simple time-stepping single simulation.

    """

    def __init__(self, model, cfun, net, spec=None, comp=None):
        self.model = model
        self.cfun = cfun
        self.net = net
        # setup code generators
        model_cg = ModelGen1(model)
        cfun_cg = CfunGen1(cfun)
        step_cg = EulerSchemeGen(model_cg)
        net_cg = NetGen1(net)
        # generate code (less than ideal api)
        spec = spec or BaseSpec(openmp=True)
        comp = comp or Compiler(openmp=True)
        comp.cflags += '-O3 -ffast-math -march=native'.split()
        net_c = net_cg.generate_c(cfun_cg, spec)
        net_cg.func.fn = getattr(comp('net', net_c), net_cg.kernel_name)
        net_cg.func.annot_types(net_cg.func.fn)
        step_c = step_cg.generate_c(spec)
        step_cg.func.fn = getattr(comp('step', step_c), step_cg.kernel_name)
        step_cg.func.annot_types(step_cg.func.fn)
        self.net_cg = net_cg
        self.step_cg = step_cg
        self.spec = spec
        self.logger = getLogger(self.__class__.__name__)

    def prep_data(self, weights):
        self.weights = weights.astype(self.spec.np_dtype)
        self.nnode = weights.shape[0]
        self.nblock = int(self.nnode / self.spec.width)
        self.arrs = self.model.prep_arrays(self.nblock, self.spec)
        self.logger.debug('nblock %d', self.nblock)
        self.logger.debug('array shape %r', self.arrs[0].shape)

    def run(self, n_iter=100):
        dt = self.model.dt
        x, i, p, f, g, o = self.arrs
        with timer():
            for _ in range(n_iter):
                self.net_cg.func(self.nnode, self.weights, i, o)
                self.step_cg.func(dt, self.nnode, x, i, p, f, g, o)
