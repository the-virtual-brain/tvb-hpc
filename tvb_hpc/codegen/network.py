from .base import BaseCodeGen, BaseSpec
from .cfun import CfunGen1
from ..network import DenseNetwork


class NetGen1(BaseCodeGen):
    """
    Provides basic code generation for networks.

    """

    template = """
{cfun_code}

void {name}(unsigned int nnode,
            {float} * __restrict weights,
            {float} * __restrict input,
            {float} * __restrict obsrv
            )
{{
    unsigned int i, j;
    {outer_loop_pragma}
    for (i=0; i<nnode; i++)
    {{
        {inner_loop_pragma}
        for (j=0; j<nnode; j++)
        {{
            {accs}
        }}
        {posts}
    }}
}}
"""

    acc_template = "{acc} += {wij}*{cfun_pre}({pre_syn}, {post_syn});"

    post_template = "{post} = {cfun_post}({post}{norm});"

    def __init__(self, net: DenseNetwork):
        self.net = net
        assert isinstance(self.net, DenseNetwork)

    @property
    def kernel_name(self):
        return 'tvb_network'

    def generate_acc(self, spec, input, obsrv, i, fname):
        nvar = self.net.model.state_sym.size
        from_idx_expr = spec.layout.generate_idx_expr('j', i, nvar)
        to_idx_expr = spec.layout.generate_idx_expr('i', i, nvar)
        return self.acc_template.format(
            wij='weights[i*nnode + j]',
            acc='%s[%s]' % (input, to_idx_expr),
            pre_syn='%s[%s]' % (obsrv, from_idx_expr),
            post_syn='%s[%s]' % (obsrv, to_idx_expr),
            cfun_pre=fname)

    def generate_post(self, spec, input, i, fname):
        nvar = self.net.model.state_sym.size
        to_idx_expr = spec.layout.generate_idx_expr('i', i, nvar)
        norm = ''
        if self.net.cfun.stat == 'mean':
            norm = ' / nnode'
        return self.post_template.format(
            post='%s[%s]' % (input, to_idx_expr),
            norm=norm,
            cfun_post=fname)

    def loop_pragmas(self, spec):
        inner, outer = '', ''
        if spec.openmp:
            inner = '#pragma omp parallel for'
            outer = '#pragma omp simd'
        return inner, outer

    def generate_code(self, cfcg: CfunGen1, spec: BaseSpec):
        cfun = self.net.cfun
        cfun_code = cfcg.generate_code(spec)
        accs = []
        posts = []
        for i, (obsrv, pre, post, input) in enumerate(cfun.io):
            accs.append(
                self.generate_acc(
                    spec, 'input', 'obsrv', i,
                    cfcg.pre_expr_kname[str(pre)]))
            posts.append(
                self.generate_post(
                    spec, 'input',  i,
                    cfcg.post_expr_kname[str(post)]))
        inner_loop_pragma, outer_loop_pragma = self.loop_pragmas(spec)
        code = self.template.format(
            cfun_code=cfun_code,
            name=self.kernel_name,
            accs='\n            '.join(accs),
            posts='\n        '.join(posts),
            inner_loop_pragma=inner_loop_pragma,
            outer_loop_pragma=outer_loop_pragma,
            **spec.dict
        )
        return code
