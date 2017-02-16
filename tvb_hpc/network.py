

from tvb_hpc.model import BaseModel
from tvb_hpc.codegen import BaseCodeGen, BaseSpec
from tvb_hpc.coupling import BaseCoupling


class DenseNetwork(BaseCodeGen):
    """
    Simple dense weights network, no delays etc.

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
        {float} post_syn = state[i/{width}*{width}*{nsvar} + {isvar}*{width} + i % {width}];
        for (j=0; j<nnode; j++)
        {{
            {float} pre_syn = state[j/{width}*{width}*{nsvar} + {isvar}*{width} + j % {width}];
            {float} el = {cfun_pre_sum}(pre_syn, post_syn);
            acc += el * weights[i*nnode + j];
        }}
        {maybe_apply_mean}
        unsigned int idx = i/{width}*{width}*{nsvar} + {isvar}*{width} + i % {width};
        input[idx] = {cfun_post_sum}(acc);
    }}
}}
"""

    def __init__(self, model: BaseModel, cfun: BaseCoupling):
        self.model = model
        self.cfun = cfun

    @property
    def kernel_name(self):
        return 'tvb_network'

    def generate_code(self, spec: BaseSpec):
        code = self.template.format(
            cfun_code=self.cfun.generate_code(spec),
            cfun_pre_sum=self.cfun._pre_sum_name,
            cfun_post_sum=self.cfun._post_sum_name,
            name=self.kernel_name,
            maybe_apply_mean='acc /= nnode;' if self.cfun.stat == 'mean' else '',
            # TODO generalize from cfun definition
            nsvar=len(self.model.state_sym),
            isvar=0,
            **spec.dict
        )
        return code

