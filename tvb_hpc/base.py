

"""
Base classes.

"""


from typing import List, Union, Dict
from numpy import dtype
from loopy import (TargetBase, LoopKernel, make_kernel, add_dtypes,
    make_reduction_inames_unique, )


# list of Loopy instructions
Isns = List[str]


class BaseKernel:

    def kernel(self, target: TargetBase, typed: bool=True) -> LoopKernel:
        "Build and return loop kernel."
        domains = self.kernel_domains()
        body = '\n'.join(self.kernel_isns())
        data = self.kernel_data()
        knl = make_kernel(domains, body, data, target=target)
        knl = make_reduction_inames_unique(knl)
        if typed:
            dtypes = self.kernel_dtypes()
            knl = add_dtypes(knl, dtypes)
        return knl

    def kernel_domains(self) -> str:
        "Return loop domains of kernel."
        return ''

    def kernel_data(self) -> List[str]:
        "Return arguments / data to kernel."
        return [key for key in self.kernel_dtypes().keys()]

    def kernel_dtypes(self) -> Dict[str, dtype]:
        "Return map of identifiers to Numpy dtypes."
        return {}

    def kernel_isns(self) -> Isns:
        "Return list of loopy instructions."
        return []
