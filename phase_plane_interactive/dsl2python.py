from models import G2DO
from mako.template import Template 

import argparse

parser = argparse.ArgumentParser(description='Demo Python code generator')
parser.add_argument('--plot', default=False, action='store_true',
                    help='launch a phase plane interactive plot')
args = parser.parse_args()


template = Template(filename='template.py')

model_str = template.render(  name='Generic2D', 
                        const=G2DO.const,
                        limit=G2DO.limit, 
                        sv=G2DO.state,
                        drift=G2DO.drift, 
                        input=G2DO.input.split() ) 
print(model_str)

if args.plot:
    from ppi import PhasePlaneInteractive
    exec(model_str)
    model = Generic2D()
    ppi = PhasePlaneInteractive(model)
    ppi()
