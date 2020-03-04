from tvb.basic.neotraits.api import HasTraits, Attr, NArray, Range, Int, Float
import numpy
import abc

# Noise class
class Noise(HasTraits):

    def __init__(self, **kwargs):
                    
        ntau = Float(
            label=":math:`ntau`",
            required=True,
            default=0.1,
            doc = """The noise correlation time"""
        )
        self.ntau = ntau.default
                     
        noise_seed = Int(
            default=42,
            doc = """A random seed used to initialise the random_stream if it is missing."""
        )
        self.noise_seed = noise_seed.default
                     
        random_stream = Attr(
            field_type=numpy.random.RandomState,
            label=":math:`random_stream`",
            # defaults to super init
            doc = """An instance of numpy's RandomState associated with this             specific Noise object. Used when you need to resume a simulation from a state saved to disk"""
        )
        self.random_stream = random_stream.default
                
        super(Noise, self).__init__(**kwargs)
        if self.random_stream is None:
            # setting the seed with 42 will always return same randomstream
            self.random_stream = numpy.random.RandomState()
            # self.random_stream = numpy.random.RandomState(self.noise_seed)

        self.dt = None
        # For use if coloured
        self._E = None
        self._sqrt_1_E2 = None
        self._eta = None
        self._h = None


    def configure(self):
        super(Noise, self).configure()

    def __str__(self):
        return simple_gen_astr(self, 'dt ntau')

    def configure_white(self, dt, shape=None):
        """Set the time step (dt) of noise or integration time"""
        self.dt = dt
        # self.log.info('White noise configured with dt=%g', self.dt)

    def configure_coloured(self, dt, shape):

        self.dt = dt
        self._E = numpy.exp(-self.dt / self.ntau)
        self._sqrt_1_E2 = numpy.sqrt((1.0 - self._E ** 2))
        self._eta = self.random_stream.normal(size=shape)
        self._dt_sqrt_lambda = self.dt * numpy.sqrt(1.0 / self.ntau)
        # self.log.info('Colored noise configured with dt=%g E=%g sqrt_1_E2=%g eta=%g & dt_sqrt_lambda=%g',
        #               self.dt, self._E, self._sqrt_1_E2, self._eta, self._dt_sqrt_lambda)


    def generate(self, shape, lo=-1.0, hi=1.0):
        "Generate noise realization."
        if self.ntau > 0.0:
            noise = self.coloured(shape)
        else:
            noise = self.white(shape)
        return noise


    def coloured(self, shape):
        "Generate colored noise. [FoxVemuri_1988]_"
        self._h = self._sqrt_1_E2 * self.random_stream.normal(size=shape)
        self._eta = self._eta * self._E + self._h
        return self._dt_sqrt_lambda * self._eta


    def white(self, shape):
        "Generate white noise."
        noise = numpy.sqrt(self.dt) * self.random_stream.normal(size=shape)
        return noise

    @abc.abstractmethod
    def gfun(self, state_variables):
        pass
    
class Additive(Noise):

    def __init__(self):
        super().__init__()            
        nsig = NArray(
            label=":math:`nsig`",
            default=numpy.array([1.0]),
            domain = Range(lo=0.0, hi=10.0, step=0.1),
            doc = """The noise dispersion, it is the standard deviation of the distribution from which the Gaussian random variates are drawn. NOTE: Sensible values are typically ~gt 1% of the dynamic range of a Model's state variables."""
        )
        self.nsig = nsig.default
                
    def gfun(self, state_variables):

        g_x = numpy.sqrt(2.0 * self.nsig)
        return g_x
    
class Multiplicative(Noise):

    def __init__(self):
        super().__init__()            
        nsig = NArray(
            label=":math:`nsig`",
            default=numpy.array([1.0]),
            domain = Range(lo=0.0, hi=10.0, step=0.1),
            doc = """The noise dispersion, it is the standard deviation of the distribution from which the Gaussian random variates are drawn. NOTE: Sensible values are typically ~gt 1% of the dynamic range of a Model's state variables."""
        )
        self.nsig = nsig.default
                    
        b = Attr(
            field_type=equations.TemporalApplicableEquation,
            label=":math:`b`",
            # defaults to super init
            doc = """A function evaluated on the state-variables, the result of which enters as the diffusion coefficient."""
        )
        self.b = b.default
                
    def gfun(self, state_variables):

        g_x = numpy.sqrt(2.0 * self.nsig * self.b)
        return g_x
    
                                                                            