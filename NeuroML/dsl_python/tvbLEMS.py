#! /usr/bin/python

import xml.etree.ElementTree as xe
import sys
# from lems.model.model import Model
# from lems.api import Model
import inspect
import filecmp
sys.path.insert(0, '/home/michiel/Documents/TVB/dsl_datafitting/lems/')
from model.model import Model
i=0

def build_constants():
    """
    Builds the constants section for the cuda kernel
    @raise ValueError: Raised when the string is not in kuramoto_network_template.c.
    """

    datatowrite.insert(num, datatowrite.pop(i).replace("<TYPE_1>",
                                                       modelist[0].parameters['global_coupling'].dimension))
    datatowrite.insert(num, datatowrite.pop(i).replace("<PARAMETER_1>",
                                                       modelist[0].parameters['global_coupling'].name))
    datatowrite.insert(num, datatowrite.pop(i).replace("<TYPE_2>",
                                                       modelist[0].parameters['global_speed'].dimension))
    datatowrite.insert(num, datatowrite.pop(i).replace("<PARAMETER_2>",
                                                       modelist[0].parameters['global_speed'].name))
    datatowrite.insert(num, datatowrite.pop(i).replace("<RECN_NAME>",
                                                       modelist[0].constants['rec_n'].name))
    datatowrite.insert(num, datatowrite.pop(i).replace("<RECN_VALUE>",
                                                       modelist[0].constants['rec_n'].value))
    datatowrite.insert(num, datatowrite.pop(i).replace("<RECSPEED_NAME>",
                                                       modelist[0].constants['rec_speed_dt'].name))
    datatowrite.insert(num, datatowrite.pop(i).replace("<RECSPEED_VALUE>",
                                                       modelist[0].constants['rec_speed_dt'].value))
    datatowrite.insert(num, datatowrite.pop(i).replace("<OMEGA_NAME>",
                                                       modelist[0].constants['omega'].name))
    datatowrite.insert(num, datatowrite.pop(i).replace("<OMEGA_VALUE>",
                                                       modelist[0].constants['omega'].value))
    datatowrite.insert(num, datatowrite.pop(i).replace("<NOISE_NAME>",
                                                       modelist[0].constants['sig'].name))
    datatowrite.insert(num, datatowrite.pop(i).replace("<NOISE_VALUE>",
                                                       modelist[0].constants['sig'].value))

def build_modeldyn():
    """
    Builds the model dynamics section for the cuda kernel
    Noise is either on or off set by the noiseOn/noiseOff select option in the xmlmodel
    @raise ValueError: Raised when the string is not in kuramoto_network_template.c.
    """

    global i

    dyn_noise = model.component_types['Kuramoto'].dynamics.derived_variables['MODEL_DYNAMICS'].select

    datatowrite.insert(num, datatowrite.pop(i).replace("<MODEL_STATE>",
                                                    modelist[0].dynamics.state_variables['MODEL_STATE'].dimension))
    datatowrite.insert(num, datatowrite.pop(i).replace("<INTEGRATION_SCHEME>",
                                                    modelist[0].dynamics.time_derivatives['INTEGRATION_SCHEME'].value))
    datatowrite.insert(num, datatowrite.pop(i).replace("<MODEL_DYNAMICS>",
                                                    modelist[0].dynamics.derived_variables['MODEL_DYNAMICS'].value))
    if dyn_noise == 'noiseON':
        datatowrite.insert(num, datatowrite.pop(i).replace("<NOISE_PARAM>",
                                                    modelist[0].constants['sig'].name))
    if dyn_noise == 'noiseOFF' and '<NOISE_PARAM>' in line:
        datatowrite.pop(i)
        datatowrite.pop(i - 1)
        i -= 2
    datatowrite.insert(num, datatowrite.pop(i).replace("<MODEL_LIMIT>",
                                                    modelist[0].requirements['MODEL_LIMIT'].dimension))
    datatowrite.insert(num, datatowrite.pop(i).replace("<OBSERV_OUTPUT>",
                                                    modelist[0].exposures['OBSERV_OUTPUT'].dimension))


def build_coupling():
    """
    Builds the model couling section for the cuda kernel
    @raise ValueError: Raised when the string is not in kuramoto_network_template.c.
    """

    # switch reduce for the actual csyntax
    switcher = {
        'add': '+=',
        'sub': '-=',
        'mul': '*=',
        'div': '/='
    }

    datatowrite.insert(num, datatowrite.pop(i).replace("<COUPLING_FUNCTION_NAME>",
                                                modelist[0].dynamics.derived_variables['COUPLING_FUNCTION'].exposure))
    datatowrite.insert(num, datatowrite.pop(i).replace("<COUPLING_FUNCTION_REDUCE>",switcher.get(
                                                modelist[0].dynamics.derived_variables['COUPLING_FUNCTION']
                                                    .reduce, "reduce argument not listed")))
    datatowrite.insert(num, datatowrite.pop(i).replace("<COUPLING_FUNCTION_VALUE>",
                                                modelist[0].dynamics.derived_variables['COUPLING_FUNCTION'].value))

def comparefiles():

    # print(filecmp.cmp(fp_cuda, fp_golden, shallow = False))
    import difflib
    import sys
    from termcolor import colored, cprint

    text1 = open(fp_cuda).readlines()
    text2 = open(fp_golden).readlines()

    num=0
    for num, line in enumerate(difflib.unified_diff(text1, text2),0):
        print(line)

    if num == 0:
        toprint=('\n The file ' + fp_cuda + ' equals ' + fp_golden + ' dead on!')
        cprint(toprint, 'cyan')

if __name__ == '__main__':

    fp_xml = '../NeuroML/RateBased_kura.xml'
    fp_templ = '../models/CUDA/kuramoto_network_template.c'
    fp_cuda = '../models/CUDA/rateB_kuramoto.c'
    fp_golden = '../models/CUDA/kuramoto_network.c'

    datatowrite=[]
    with open(fp_templ, 'r') as c:
        datatoread = c.readlines()

    model = Model()
    model.import_from_file(fp_xml)

    modelist=list()
    modelist.append(model.component_types['Kuramoto'])



    # print(inspect.getmembers(modelist[0]))
    print(vars(modelist[0].dynamics.derived_variables['COUPLING_FUNCTION']))

    a = modelist[0].dynamics.derived_variables['COUPLING_FUNCTION']
    print(a.__dict__)

    for num, line in enumerate(datatoread,1):
        datatowrite.append(line)
        build_constants()
        build_modeldyn()
        build_coupling()
        i += 1

    with open(fp_cuda, "w") as f:
        f.writelines(datatowrite)

    #compare files
    comparefiles()

    print("----------------------------------------------")