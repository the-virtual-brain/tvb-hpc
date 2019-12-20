from tvb.basic.neotraits.api import NArray, Final, List, Range, HasTraits
import numpy

# Coupling class
class Difference(HasTraits):

    r"""
    Provides a difference coupling function, between pre and post synaptic
    activity of the form
    .. math::
        a G_ij (x_j - x_i)
    """

    def __init__(self):
            
        a = NArray(
            label=":math:`a`",
            default=numpy.array([0.1]),
            domain = Range(lo=0.0, hi=10., step=0.1),
            doc = """Rescales the connection strength."""
        )
        self.a = a
        
    def __str__(self):
        return simple_gen_astr(self, a)

    def pre(self, V, V_j):
        return sin(V_j - V)

    def post(self, wij):
        a = self.a.default
        return wij


                                                        