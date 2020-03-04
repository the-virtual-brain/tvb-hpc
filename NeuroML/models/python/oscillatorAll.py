from tvb.basic.neotraits.api import NArray, Final, List, Range, HasTraits
import numpy

class Oscillator:

    def __init__(self):

    # Define traited attributes for this model, these represent possible kwargs.
            
        I = NArray(
            label=":math:`I`",
            default=numpy.array([0.0]),
            domain = Range(lo=-5.0, hi=5.0, step=0.01),
            doc = """Baseline shift of the cubic nullcline"""
        )
        self.I = I
                    
        tau = NArray(
            label=":math:`tau`",
            default=numpy.array([1.0]),
            domain = Range(lo=1.0, hi=5.0, step=0.01),
            doc = """A time-scale hierarchy can be introduced for the state variables :math:`V` and :math:`W`. Default parameter is 1, which means no time-scale hierarchy."""
        )
        self.tau = tau
                    
        a = NArray(
            label=":math:`a`",
            default=numpy.array([-2.0]),
            domain = Range(lo=-5.0, hi=5.0, step=0.01),
            doc = """Vertical shift of the configurable nullcline"""
        )
        self.a = a
                    
        b = NArray(
            label=":math:`b`",
            default=numpy.array([-10.0]),
            domain = Range(lo=-20.0, hi=15.0, step=0.01),
            doc = """Linear slope of the configurable nullcline"""
        )
        self.b = b
                    
        c = NArray(
            label=":math:`c`",
            default=numpy.array([0]),
            domain = Range(lo=-10.0, hi=10.0, step=0.01),
            doc = """Parabolic term of the configurable nullcline"""
        )
        self.c = c
                    
        d = NArray(
            label=":math:`d`",
            default=numpy.array([0.02]),
            domain = Range(lo=0.0001, hi=1.0, step=0.0001),
            doc = """Temporal scale factor. Warning: do not use it unless you know what you are doing and know about time tides."""
        )
        self.d = d
                    
        e = NArray(
            label=":math:`e`",
            default=numpy.array([3.0]),
            domain = Range(lo=-5.0, hi=5.0, step=0.0001),
            doc = """Coefficient of the quadratic term of the cubic nullcline."""
        )
        self.e = e
                    
        f = NArray(
            label=":math:`f`",
            default=numpy.array([1.0]),
            domain = Range(lo=-5.0, hi=5.0, step=0.0001),
            doc = """Coefficient of the cubic term of the cubic nullcline."""
        )
        self.f = f
                    
        g = NArray(
            label=":math:`g`",
            default=numpy.array([0.0]),
            domain = Range(lo=-5.0, hi=5.0, step=0.5),
            doc = """Coefficient of the linear term of the cubic nullcline."""
        )
        self.g = g
                    
        alpha = NArray(
            label=":math:`alpha`",
            default=numpy.array([1.0]),
            domain = Range(lo=-5.0, hi=5.0, step=0.0001),
            doc = """Constant parameter to scale the rate of feedback from the slow variable to the fast variable."""
        )
        self.alpha = alpha
                    
        beta = NArray(
            label=":math:`beta`",
            default=numpy.array([1.0]),
            domain = Range(lo=-5.0, hi=5.0, step=0.0001),
            doc = """Constant parameter to scale the rate of feedback from the slow variable to itself"""
        )
        self.beta = beta
                    
        gamma = NArray(
            label=":math:`gamma`",
            default=numpy.array([1.0]),
            domain = Range(lo=-1.0, hi=1.0, step=0.1),
            doc = """Constant parameter to reproduce FHN dynamics where excitatory input currents are negative. It scales both I and the long range coupling term.."""
        )
        self.gamma = gamma
        
        state_variable_range = Final(
            label="State Variable ranges [lo, hi]",
            default={    "V": numpy.array([[-2.0, 4.0]]), 
				     "W": numpy.array([[-6.0, 6.0]])},
            doc="""V"""
        )

        state_variables = ('V', 'W')

        _nvar = 2
        cvar = numpy.array([0], dtype=numpy.int32)

        self.I = I
        self.tau = tau
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.state_variables = state_variables
        self.state_variable_range = state_variable_range

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0):

        V = state_variables[0, :]
        W = state_variables[1, :]

        #[State_variables, nodes]
        # c_0 = coupling[0, :]
        c_0 = coupling

        # TODO why does it not default auto to default
        I = self.I.default
        tau = self.tau.default
        a = self.a.default
        b = self.b.default
        c = self.c.default
        d = self.d.default
        e = self.e.default
        f = self.f.default
        g = self.g.default
        alpha = self.alpha.default
        beta = self.beta.default
        gamma = self.gamma.default

        lc_0 = local_coupling * V
        derivative = numpy.empty_like(state_variables)

        # TODO fixed the acceptance of ** but it will process *** now as well. However not as an operand but as a value or node
        derivative[0] = d * tau * (alpha * W - f * V**3 + e * V**2 + g * V + gamma * I + gamma * c_0 + lc_0)
        derivative[1] = d * (a + b * V + c * V**2 - beta * W) / tau

        return derivative

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
        
        self.a = a

    def __str__(self):
        return simple_gen_astr(self, a)

    def pre(self, x_i, x_j):
        return x_j - x_i

    def post(self, gx):
        a = self.a.default
        return a * gx



# Noise class
class Noise(HasTraits):

    def __init__(self, **kwargs):
            
        ntau = Float(
            label=":math:`ntau`",
            required=nconst.dimension,
            default=0.0,
            doc = """The noise correlation time"""
        )
        self.ntau = ntau
                     
        noise_seed = Int(
            default=42,
            doc = """A random seed used to initialise the random_stream if it is missing."""
        )
        self.noise_seed = noise_seed
                     
        random_stream = Attr(
            field_type=numpy.random.RandomState,
            label=":math:`random_stream`",
            # defaults to super init
            doc = """An instance of numpy's RandomState associated with this             specific Noise object. Used when you need to resume a simulation from a state saved to disk"""
        )
        self.random_stream = random_stream
                
        super(Noise, self).__init__(**kwargs)
        if self.random_stream is None:
            self.random_stream = numpy.random.RandomState(self.noise_seed)

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
        self.log.info('White noise configured with dt=%g', self.dt)

    def configure_coloured(self, dt, shape):

        self.dt = dt
        self._E = numpy.exp(-self.dt / self.ntau)
        self._sqrt_1_E2 = numpy.sqrt((1.0 - self._E ** 2))
        self._eta = self.random_stream.normal(size=shape)
        self._dt_sqrt_lambda = self.dt * numpy.sqrt(1.0 / self.ntau)
        self.log.info('Colored noise configured with dt=%g E=%g sqrt_1_E2=%g eta=%g & dt_sqrt_lambda=%g',
                      self.dt, self._E, self._sqrt_1_E2, self._eta, self._dt_sqrt_lambda)


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
            
        nsig = NArray(
            label=":math:`nsig`",
            default=numpy.array([1.0]),
            domain = Range(lo=0.0, hi=10.0, step=0.1),
            doc = """The noise dispersion, it is the standard deviation of the distribution from which the Gaussian random variates are drawn. NOTE: Sensible values are typically ~gt 1% of the dynamic range of a Model's state variables."""
        )
        self.nsig = nsig
                
    def gfun(self, state_variables):

        g_x = numpy.sqrt(2.0 * self.nsig)
        return g_x
    
class Multiplicative(Noise):

    def __init__(self):
            
        nsig = NArray(
            label=":math:`nsig`",
            default=numpy.array([1.0]),
            domain = Range(lo=0.0, hi=10.0, step=0.1),
            doc = """The noise dispersion, it is the standard deviation of the distribution from which the Gaussian random variates are drawn. NOTE: Sensible values are typically ~gt 1% of the dynamic range of a Model's state variables."""
        )
        self.nsig = nsig
                    
        b = Attr(
            field_type=equations.TemporalApplicableEquation,
            label=":math:`b`",
            # defaults to super init
            doc = """A function evaluated on the state-variables, the result of which enters as the diffusion coefficient."""
        )
        self.b = b
                
    def gfun(self, state_variables):

        g_x = numpy.sqrt(2.0 * self.nsig * self.b)
        return g_x
    
                                                                            \