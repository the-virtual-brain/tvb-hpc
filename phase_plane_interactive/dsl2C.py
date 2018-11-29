import ast

from ctree.c.nodes import *
from ctree.transformations import PyBasicConversions

# for future use, not needed now..
#index = ('i','j', 1)
#dims = ('n_i', 'n_j', 3)
#print(gen_index_arithmetic(index,dims))
## i*n_j*3 + j*3 + 1
def gen_index_arithmetic(index,dims):
    s = []
    for i in range(len(dims)):
        tree = SymbolRef(index[i])
        for j in range(i+1, len(dims)):
            tree = BinaryOp( left=tree,
                             right=SymbolRef(dims[j]),
                             op=Op.Mul() )
        s.append(tree)
    tree = s[0]
    for i in range(1,len(s)):
        tree = BinaryOp( left=tree,
                         right=s[i],
                         op=Op.Add() )
    return tree

class PyBasicConversionsExtended(PyBasicConversions):
    def visit_BinOp(self,node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if type(node.op) == ast.Pow:
            return FunctionCall(func=SymbolRef('pow'),args=[lhs,rhs])
        else:
            op = self.PY_OP_TO_CTREE_OP.get(type(node.op), type(node.op))()
            return BinaryOp(lhs, op, rhs)


def dfuns_to_c(dfuns):
    c_dfuns = []
    for dfun in dfuns:
        dfun_ast = ast.parse(G2DO.drift[0]).body[0]
        c_dfuns.append( PyBasicConversionsExtended().visit(dfun_ast).codegen())
    return c_dfuns
        

if __name__=='__main__':


    from ppi import G2DO
    from mako.template import Template

    template = Template(filename='cuda_model.cu')

    print(template.render(  svar=G2DO.state,const=G2DO.const, cvar=G2DO.input,
                            dfuns=dfuns_to_c(G2DO.drift)))
    

