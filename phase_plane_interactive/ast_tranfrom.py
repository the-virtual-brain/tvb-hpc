import ast 
import astor


class IndexArithmetic(ast.NodeTransformer):
    def __init__(self, index_map):
        self.index_map = index_map

    def visit_Name(self, node):
        if node.id not in self.index_map:
            return node
        index = self.index_map[node.id]

        new_node = ast.Subscript(
            value=ast.Name(id='X', ctx=ast.Load()),
            slice=ast.Index(value=index),
            ctx=node.ctx
        )
        return new_node

def index_dfun(model):

    n_svar = len(model.state)
    svar_i = dict(zip(model.state,range(n_svar)))
    
    dfun_asts = []
    for drift in model.drift:
        dfun_asts.append(ast.parse(drift).body[0].value)


    svar_index = dict.fromkeys(model.state)

    for i, var in enumerate(model.state):
        index = ast.Tuple(
                ctx=ast.Load(),
                elts=[  ast.Num(n=i),
                        ast.Name(id='t', ctx=ast.Load())
                    ])
        svar_index[var] = index

    dfun_idx_asts = []
    for i, var in enumerate(model.state):
         
        index = svar_index[var]
        dX = ast.Subscript(
                value=ast.Name(id='dX', ctx=ast.Load()),
                slice=ast.Index(value=index),
                ctx=ast.Store() )
        dfun_idx_ast = ast.Assign(
                targets=[dX],
                value=IndexArithmetic(svar_index).generic_visit(dfun_asts[i]))
        dfun_idx_asts.append( dfun_idx_ast )

    nowrap = lambda x:''.join(x)
    dfun_strings = list(map(lambda x: astor.to_source(x, pretty_source=nowrap), 
                            dfun_idx_asts ) )

    return dfun_strings


if __name__=="__main__":
    from ppi import G2DO
    from mako.template import Template

    dfuns = index_dfun(G2DO)

    template = Template(filename='numba_model.py')
    print(template.render(dfuns=dfuns, const=G2DO.const, cvar=G2DO.input))
