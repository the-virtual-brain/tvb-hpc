from ppi import G2DO
from mako.template import Template 

template = Template(filename='template.py')
print(template.render(  name='Generic2D', 
                        const=G2DO.const,
                        limit=G2DO.limit, 
                        sv=G2DO.state.split(),
                        drift=G2DO.drift, 
                        input=G2DO.input.split() ) )
