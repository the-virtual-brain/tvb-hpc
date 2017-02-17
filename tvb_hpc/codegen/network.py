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
            {float} * __restrict state,
            {float} * __restrict input
            )
{{
    unsigned int i, j;
    for (i=0; i<nnode; i++)
    {{
        {float} acc = 0;
        {float} post_syn = state[i/{width}*{width}*{nsvar} \\
                                 + {isvar}*{width} + i % {width}];
        for (j=0; j<nnode; j++)
        {{
            {float} pre_syn = state[j/{width}*{width}*{nsvar} \\
                                    + {isvar}*{width} + j % {width}];
            {float} el = {cfun_pre_sum}(pre_syn, post_syn);
            acc += el * weights[i*nnode + j];
        }}
        {maybe_apply_mean}
        unsigned int idx = i/{width}*{width}*{nsvar} \\
                                + {isvar}*{width} + i % {width};
        input[idx] = {cfun_post_sum}(acc);
    }}
}}
"""

    def __init__(self, net: DenseNetwork):
        self.net = net
        assert isinstance(self.net, DenseNetwork)

    @property
    def kernel_name(self):
        return 'tvb_network'

    def generate_code(self, cfcg: CfunGen1, spec: BaseSpec):
        maybe_apply_mean = ''
        if self.net.cfun.stat == 'mean':
            maybe_apply_mean = 'acc /= nnode;'
        code = self.template.format(
            cfun_code=cfcg.generate_code(spec),
            cfun_pre_sum=cfcg.kernel_name_pre_sum,
            cfun_post_sum=cfcg.kernel_name_post_sum,
            name=self.kernel_name,
            maybe_apply_mean=maybe_apply_mean,
            # TODO generalize from cfun definition
            nsvar=len(self.net.model.state_sym),
            isvar=0,
            **spec.dict
        )
        return code
