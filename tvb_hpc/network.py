from tvb_hpc.model import BaseModel
from tvb_hpc.coupling import BaseCoupling


class DenseNetwork:
    """
    Simple dense weights network, no delays etc.

    """

    def __init__(self, model: BaseModel, cfun: BaseCoupling):
        self.model = model
        self.cfun = cfun
