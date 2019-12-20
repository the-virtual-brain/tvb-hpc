from tvb.basic.neotraits.api import NArray, Final, List, Range, HasTraits
import numpy

class ${dfunname}:

    def __init__(self):

    # Define traited attributes for this model, these represent possible kwargs.
    %for mconst in const:
        %if mconst.symbol == 'NArray':
            ${NArray(mconst)}\
        %elif mconst.symbol == 'Attr':
            ${Attr(mconst)}\
        %elif mconst.symbol == 'Float':
            ${Float(mconst)} \
        %elif mconst.symbol == 'Int':
            ${Int(mconst)} \
        %endif
    %endfor

        state_variable_range = Final(
            label="State Variable ranges [lo, hi]",
            default={\
    %for item in dynamics.state_variables:
    "${item.name}": numpy.array([${item.dimension}])${'' if loop.last else ', \n\t\t\t\t '}\
    %endfor
},
            doc="""${dynamics.state_variables['V'].exposure}"""
        )

        state_variables = (\
%for item in dynamics.state_variables:
'${item.name}'${'' if loop.last else ', '}\
%endfor
)

        _nvar = ${dynamics.state_variables.__len__()}
        cvar = numpy.array([0], dtype=numpy.int32)

        % for item in const:
        self.${item.name} = ${item.name}
        % endfor
        self.state_variables = state_variables
        self.state_variable_range = state_variable_range

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0):

        % for i, item in enumerate(dynamics.state_variables):
        ${item.name} = state_variables[${i}, :]
        % endfor

        #[State_variables, nodes]
        # c_0 = coupling[0, :]
        c_0 = coupling

        # TODO why does it not default auto to default
        %for item in const:
        ${item.name} = self.${item.name}.default
        %endfor

        lc_0 = local_coupling * V
        derivative = numpy.empty_like(state_variables)

        # TODO fixed the acceptance of ** but it will process *** now as well. However not as an operand but as a value or node
        % for i, item in enumerate(dynamics.time_derivatives):
        derivative[${i}] = ${item.value}
        % endfor

        return derivative

# Coupling class
class ${couplingname}(HasTraits):

    r"""
    Provides a difference coupling function, between pre and post synaptic
    activity of the form
    .. math::
        a G_ij (x_j - x_i)
    """

    def __init__(self):
    %for cconst in couplingconst:
        %if cconst.symbol == 'NArray':
            ${NArray(cconst)}\
        %elif cconst.symbol == 'Attr':
            ${Attr(cconst)}\
        %elif cconst.symbol == 'Float':
            ${Float(cconst)} \
        %elif cconst.symbol == 'Int':
            ${Int(cconst)} \
        %endif

        self.${cconst.name} = ${cconst.name}
     %endfor

    def __str__(self):
        return simple_gen_astr(self, \
%for item in couplingconst:
${item.name})
% endfor

    %for item in couplingfunctions:
    def ${item.name}(self, \
        % if item.name == 'pre':
            %for param in couplingparams:
${param.name}${'' if loop.last else ', '}\
            % endfor
):
        return ${item.value}

        % elif item.name == 'post':
            % for reqs in couplingreqs:
${reqs.name}${'' if loop.last else ', '}\
            % endfor
):
        %for cnst in couplingconst:
        ${cnst.name} = self.${cnst.name}.default
        % endfor
        return ${item.value}

        % endif
    % endfor


# Noise class
%for i, type in enumerate(noisetypes):
class ${noisetypes[i]}(${'HasTraits' if loop.first else noisetypes[0]}):

    def __init__(self${', **kwargs' if loop.first else ''}):
    %for nconst in noiseconst[i]:
        %if nconst.symbol == 'NArray':
            ${NArray(nconst)}\
        %elif nconst.symbol == 'Attr':
            ${Attr(nconst)}\
        %elif nconst.symbol == 'Float':
            ${Float(nconst)} \
        %elif nconst.symbol == 'Int':
            ${Int(nconst)} \
        %endif
    % endfor
    % if loop.first:
        ${printnoise(0)}
    % else:
        ${printchildren(i)}
    % endif
    % endfor
    \
    <%def name="printchildren(i)">
    def gfun(self, state_variables):

        g_x = numpy.sqrt(2.0 * \
        %for name in noiseconst[i]:
self.${name.name}${'' if loop.last else ' * '}\
        %endfor
)
        return g_x
    </%def>\
    \
    <%def name="printnoise(x)">
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
    </%def>\
    \
        ## TVB numpy constant declarations
        <%def name="NArray(nconst)">
        ${nconst.name} = ${nconst.symbol}(
            label=":math:`${nconst.name}`",
            default=numpy.array([${nconst.value}]),
            domain = Range(${nconst.dimension}),
            doc = """${nconst.description}"""
        )
        self.${nconst.name} = ${nconst.name}
        </%def>\
        \
        <%def name="Attr(nconst)">
        ${nconst.name} = ${nconst.symbol}(
            ## todo: adapt fields in LEMS to match TVBs constant requirements more closely
            field_type=${nconst.dimension},
            label=":math:`${nconst.name}`",
            # defaults to super init
            doc = """${nconst.description}"""
        )
        self.${nconst.name} = ${nconst.name}
        </%def>\
        \
        <%def name="Float(nconst)">
        ${nconst.name} = ${nconst.symbol}(
            ## todo: adapt fields in LEMS to match TVBs constant requirements more closely
            label=":math:`${nconst.name}`",
            required=nconst.dimension,
            default=${nconst.value},
            doc = """${nconst.description}"""
        )
        self.${nconst.name} = ${nconst.name}
        </%def>\
        \
        <%def name="Int(nconst)">
        ${nconst.name} = ${nconst.symbol}(
            default=${nconst.value},
            doc = """${nconst.description}"""
        )
        self.${nconst.name} = ${nconst.name}
        </%def>\