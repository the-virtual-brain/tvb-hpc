# from models import G2DO
from mako.template import Template 

import argparse

import sys
sys.path.insert(0, '/home/michiel/Documents/TVB/dsl_datafitting/lems/')
from model.model import Model

if True:
    # model specifiers
    modelname='Oscillator'
    modeldrift='drift'
    modelcoupling='Difference'

    fp_xml = '../NeuroML/' + modelname.lower() + '.xml'
    # fp_templ = '../models/CUDA/kuramoto_network_template.c'
    # fp_cuda = '../models/CUDA/rateB_kuramoto.c'
    # fp_golden = '../models/CUDA/kuramoto_network.c'

    model = Model()
    model.import_from_file(fp_xml)
    modelextended = model.resolve()

    # drift dynamics
    modelist = list()
    modelist.append(model.component_types[modeldrift])

    # coupling functionality
    couplinglist = list()
    couplinglist.append(model.component_types['coupling_function'])

    # noise collection
    noisetypes = list()
    for i, item in enumerate(model.component_types):
        if item.name == 'noise' or item.extends == 'noise':
            noisetypes.append(item.name)

    noiselist = list()
    noiseconstantspernoise = [[] for i in range(len(noisetypes))]
    for i, type in enumerate(noisetypes):
        for j, const in enumerate(model.component_types[type].constants):
            noiseconstantspernoise[i].append(const)

    noisetypes = [x.title() for x in noisetypes]
    # print((noiseconstantspernoise[0][1].name))
    #
    # for i, smtg in enumerate(noiseconstantspernoise):
    #     for j, kjks in enumerate(smtg):
    #         print(j, kjks.name)

    # for item in noisetypes:
    #     noiselist.append(modelextended.component_types[item.lower()])

    # for i, item in enumerate(noisetypes):
    #     print((noiselist[i].constants))


    # print((model.component_types['additive'].constants['ntau'].name))

    # print((model.component_types['oscillator'].constants['I'].value))

    # print(inspect.getmembers(modelist[0]))
    # print(vars(modelist[0].constants['COUPLING_FUNCTION']))
    # print((modelist[0].constants['tau'].__dict__))

    # printing the different items of the xml
    # constants = modelist[0].constants
    # for value in constants:
    #     print(value.__dict__)
    #     for k, v in value.__dict__.items():
    #         print(k,':',v)

    # constants = modelist[0].constants
    # for value in constants:
    #     print(value.value)

    # a = modelist[0].dynamics.derived_variables['COUPLING_FUNCTION']
    # print(a.__dict__)

    # parser = argparse.ArgumentParser(description='Demo Python code generator')
    # parser.add_argument('--plot', default=False, action='store_true',
    #                     help='launch a phase plane interactive plot')
    # args = parser.parse_args()

    template = Template(filename='tmpl8_L2py.py')
    model_str = template.render(
                            dfunname=modelname,
                            couplingname=modelcoupling,
                            const=modelist[0].constants,
                            couplingconst=couplinglist[0].constants,
                            dynamics=modelist[0].dynamics,
                            couplingparams=couplinglist[0].parameters,
                            couplingreqs=couplinglist[0].requirements,
                            couplingfunctions=couplinglist[0].functions,
                            noiseconst=noiseconstantspernoise,
                            # noiseconst=noiselist[0].constants,
                            noisetypes=noisetypes,

                            drift=(1e-3, 1e-3),
                            input='c_0'
                            )
    # print(model_str)

    modelfile="../models/oscillatorAll.py"
    with open(modelfile, "w") as f:
        f.writelines(model_str)

    template = Template(filename='tmpl8_drift.py')
    model_str = template.render(
                            dfunname=modelname,
                            const=modelist[0].constants,
                            dynamics=modelist[0].dynamics,
                            drift=(1e-3, 1e-3)
                            )
    # print(model_str)

    modelfile="../models/oscillator.py"
    with open(modelfile, "w") as f:
        f.writelines(model_str)

    template = Template(filename='tmpl8_coupling.py')
    model_str = template.render(
                            couplingname=modelcoupling,
                            couplingconst=couplinglist[0].constants,
                            couplingparams=couplinglist[0].parameters,
                            couplingreqs=couplinglist[0].requirements,
                            couplingfunctions=couplinglist[0].functions
                            )
    # print(model_str)

    modelfile="../models/coupling.py"
    with open(modelfile, "w") as f:
        f.writelines(model_str)

    template = Template(filename='tmpl8_noise.py')
    model_str = template.render(
        noiseconst=noiseconstantspernoise,
        # noiseconst=noiselist[0].constants,
        noisetypes=noisetypesLEMS2python.py
    )
    # print(model_str)

    modelfile = "../models/noise.py"
    with open(modelfile, "w") as f:
        f.writelines(model_str)

# fp_extest = '../NeuroML/example10_Q10.xml'
# extest = Model()
# extest.import_from_file(fp_extest)
# extest = extest.resolve()
# # print((extest.component_types['TemperatureDependency'].exposures['rateFactor'].name))
# # print((extest.component_types['Q10TemperatureDependency'].parameters['Q10'].name))
#
# # print((extest.component_types['TemperatureDependency'].parameters['Q10'].name))
# # print((extest.component_types['Q10TemperatureDependency'].requirements['velocity'].name))
# print(extest.component_types['Q10TemperatureDependency'].requirements['temperature'].name)


# if args.plot:
#     from ppi import PhasePlaneInteractive
#     exec(model_str)
#     model = Generic2D()
#     ppi = PhasePlaneInteractive(model)
#     ppi()


# %        for k, v in value.__dict__.items():
#            ${v}
#         %endfor