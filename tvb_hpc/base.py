

"""
Base classes.

"""


from typing import List, Dict
from numpy import dtype
import pymbolic as pm
import loopy as lp
from loopy import (
    TargetBase, LoopKernel, make_kernel, add_and_infer_dtypes,
    make_reduction_inames_unique, generate_code)


# list of Loopy instructions
Isns = List[str]


class BaseKernel:

    def code(self, *args, **kwargs):
        knl = self.kernel(*args, **kwargs)
        code, _ = generate_code(knl)
        return code

    def kernel(self, target: TargetBase, typed: bool=True) -> LoopKernel:
        "Build and return loop kernel."
        domains = self.kernel_domains()
        body = '\n'.join(self.kernel_isns())
        data = self.kernel_data()
        knl = make_kernel(domains, body, data, target=target)
        knl = make_reduction_inames_unique(knl)
        knl.name = self.__class__.__name__
        if typed:
            dtypes = self.kernel_dtypes()
            knl = add_and_infer_dtypes(knl, dtypes)
        return knl

    def kernel_domains(self) -> str:
        "Return loop domains of kernel."
        return self.domains

    def kernel_data(self) -> List[str]:
        "Return arguments / data to kernel."
        # normalize wrt. key set like ['n,out', 'foo,bar']
        csk = ','.join(self.kernel_dtypes().keys())
        data = [key for key in csk.split(',')]
        if hasattr(self, 'extra_data_shape'):
            for name, shape in self.extra_data_shape.items():
                shape = tuple(pm.parse(_) for _ in shape.split(','))
                arg = lp.GlobalArg(name, shape=shape)
                data[data.index(name)] = arg
        return data

    def kernel_dtypes(self) -> Dict[str, dtype]:
        "Return map of identifiers to Numpy dtypes."
        return self.dtypes

    def kernel_isns(self) -> Isns:
        "Return list of loopy instructions."
        return [_[4:] for _ in self.instructions.split('\n')]
