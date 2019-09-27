# TVB_DSL XML (LEMS) to CUDA code generation
This readme describes the usage of the automatic code generation of the XML to CUDA. As an example 'tvbLEMS.py' takes the Kuramoto model (model for the behaviour of large set of coupled ocillators) defined in 'RateBased_kura.xml' and translates this in a CUDA file called 'rateB_kura.c'. It then compares the resulting output file to the golden file 'kuramoto_network.c' and outputs any differences between the file. A unit test which compares two files can also be selected.

# Author
Sandra Diaz & Michiel van der Vlag @ Forschungszentrum Juelich.

# License
Apache License. Version 2.0, January 2004

# Files
* /dsl_python/tvbLEMS.py 			: python script to convert XML(LEMS) to CUDA
* /NeuroMl/RateBased_kura.xml 			: XML LEMS format containing Kuramoto model
* /models/CUDA/kuratmoto_network_template.c 	: Template with placeholders for XML model
* /models/CUDA/kuramoto_network.c 		: Golden model file for comparison
* /models/CUDA/rateB_kura.c 			: Resulting Cuda output file
* /unittests/validate_output_equality.py	: Unittest to validate equality of output

# Prerequisites
pyLEMS: https://github.com/LEMS/pylems. 
sudo pip install pylems

# XML LEMS Definitions 
http://lems.github.io/LEMS/elements.html
This section defines the fields on the XML file and how they have been interpretated for the Kuramoto model.

```XML
* <Parameter name="string" dimension="string"/>
	Sets the name and dimensinality of the parameter that must be supplied when a component is defined. These are the inputs from outside to the model.

* <Requirement name="string" dimension="string" description="string"/>
	Sets the name and dimensionality that should be accessible within the scope of a model component. Used for selecting the wrapping function for the limits of the model. The desciption holds the actual value.

* <Constant name="string" dimension="string" value="string"/>
	This is like a parameter but the value is supplied within the model definition itself. The dimension is used to set the unit of the constant. 

* <Exposure name="string" dimension="string"/>
	For variables that should be made available to other components. Used to output the results of the model. 

Dynamics
	Specifies the dynamical behaviour of the model.

   * <StateVariable name="string" dimension="string"/>
   		Name of the state variable in state elements. Dimension is used for the value

   * <DerivedVariable name="string" exposure="string" value="string" reduce="add/mul" select="noiseOn/noiseOff"/>
   		A quantity that depends algebraically on other quantities in the model. 
		The 'value' field can be set to a mathematical expression. 
		The reduce field is used to set the mathematical expression (Add: +=, Mul: *=). 		The select field can be used to add noise to the integration step.

   * <TimeDerivative variable="string" value="string"/>
   		Expresses the time step uped for the model. Variable is used for the name.
```


