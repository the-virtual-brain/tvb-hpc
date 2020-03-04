# TVB_DSL XML (LEMS) to CUDA code generation
This readme describes the usage of the automatic code generation of the XML LEMS ratebase models to CUDA or Python. 
'converters/LEMS2python.py' takes a LEMS model and converts it to a TVB based python model.
'converters/LEMS2CUDA.py' takes the same LEMS model and converts it to a CUDA model. 
The converters make use of the mako templating library. 
The TVB models, which can specify the model dynamics, noise and coupling functionality  can be hooked up into 
the TVB lite version to run a simulation. 

# Author
Sandra Diaz & Michiel van der Vlag @ Forschungszentrum Juelich.

# License
Apache License. Version 2.0, January 2004

# Files
* /dsl_python/*: python scripts to convert XML(LEMS) to CUDA and Python models
* /models/*: converted c and python models
* /models/NeuronML/*: ratebased models based on XML LEMS format
* /models/CUDA/*: original CUDA models
* /unittests/*: unittest to validate equality of output (depracated)
* /lems/*: adapted pyLEMS directory original by P. Gleeson
* /tvb/tvb_destilled_generic.py: tvb lite version with hooks for models, noise and coupling generated.

# Dependancies
pyLEMS (locally)
numpy
mako

# XML LEMS Definitions 
http://lems.github.io/LEMS/elements.html
This section defines the necessary fields on the XML file and how they have been interpretated for model conversion.
Each LEMS directive yields an example. There are component types for models, noise and coupling functionality.

```XML
<ComponentType name="Epileptor_2D" description="optional"/>
    Specifies the model being used. Currently Oscillator_2D, Kuramoto, rWongWang and Epileptor_2D have a LEMS file.   

    <Constant name="Iext" symbol="float32" dimension="lo=1.5, hi=5.0, step=0.1" value="3.1" description="Optional"/>
        Supplies the models constant values. 
        Name: the constant name as they would appear in the models.
        Symbol: the datatype. 
        Dimension: the range for parameter sweeps (python).
        Value: the initial values.  
    
    <Parameter name="global_coupling" dimension="float"/>
        Sets the parameters that will be sweeped. All parametes supplied here are supposed to be sweeped. 
        Name: the parameter name as they would appear in the models.
        Dimension: the datatype.
    
    <Exposure name="o_-x1+x2" dimension="-x1 + x2"/>
        For variables that should be made available to other components, these are the output observables.
        Name: the constant name as they would appear in the models.
        Dimension: the function that is performed on the observable. 
    
    <Dynamics>
        Specifies the dynamical behaviour of the model.
    
        <StateVariable name="x1" dimension="-20., 2."/>
            Lists the state variables.
            Name: the state variable name as they would appear in the models.
            Dimension: the range of the state variables.
    
        <DerivedVariable name="rec_speed_dt" exposure="float" value="1.0f / global_speed / (dt)"/>
            A quantity that depends algebraically on other quantities in the model. 
            Name: the derived variable name as they would appear in the models.
            Exposure: the datatype
            Value: field can be set to a mathematical expression. 
            
        <TimeDerivative variable="dy1" value="tt * (c - d * powf(x1, 2) - y1)"/>
            Expresses the time step used for the model. 
            Variable: the time derivative name as they would appear in the models.
            Value: the derivative function
    </Dynamics>
</ComponentType>

<ComponentType name="coupling_function_pop1">
    Should always start with coupling and is then recognized as such. Multiple coupling functions can be specified. If
    no pre or post coupling function is present no signal transmission delay coupling will take place 
    ie 'Wij * (post - pre)'. A local coupling factor for, for instance 2 population can be entered as a derived 
    parameter. 

        <Constant name="Iext" symbol="float32" dimension="lo=1.5, hi=5.0, step=0.1" value="3.1" description="Optional"/>
            Supplies the models constant values. 
            Name: the constant name as they would appear in the models.
            Symbol: the datatype. 
            Dimension: the range for parameter sweeps (python).
            Value: the initial value

        <Function name="pre" value="sin(x1_j - x1)" description="pre synaptic function for coupling activity"/>
        <Function name="post" value="a" description="post synaptic = a * pre"/>
            Name: the coupling name should always be 'pre' and 'post'.
            Value: the function as it would appear in the generated model. 
            Descripition: optional, does appear in Python but not in CUDA.

        <DerivedParameter name="c_pop2" value="g"/>
            Handle local coupling result, full expression is 'name' *= 'value'. When nog signal transmission difference
            needs to be calculated these derived parameter can still specify a function to couple populations. 
        
</ComponentType>

<ComponentType name="noise">
    For the CUDA models just entering a component type 'noise' will enable the noise function in CUDA. This is the 
    per state variable stated sigma value multiplied by the CUDA random function (curand_normal). Noise can be extended
    with additional noise features. For the Python models contants can be entered for different types of noise.
    For the numpy array float32 will be assumed. 

        <Constant name="ntau" dimension="lo=0.0, hi=10.0, step=0.1" value="0.1" description="optional"/>        

<ComponentType name="additive" extends="noise">
    Specific type of noise of which next to the constants also the funtion can be specified.

        <Constant name="nsig" dimension="lo=0.0, hi=10.0, step=0.1" value="1.0" description="optional"/>
    
        <Function name="noisefunction" value="" description="post synaptic = a * pre"/>

```

# Acknowledgement
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant 
agreement No 785907 (HBP SGA2).

# TODO
* Make LEMS directives more consistent. Entails alteration of pyLEMS
* Produce Numba based models and compare performance against CUDA version.
* Add Numba test suite for n number of models.
* Add CUDA test suite for n number of models.
* CUDA models only have curand based random noise.
* Install parameter sweep backend suite based on STAN.
* Install paramerer sweep backend based on L2L.