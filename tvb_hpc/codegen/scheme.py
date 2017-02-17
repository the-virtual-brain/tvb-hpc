
from .base import BaseCodeGen, Loop, indent, Storage


class EulerSchemeGen(BaseCodeGen):

    template = """
{model_code}

void {name}(
    {float} dt,
    unsigned int nnode,
    {float} *state,
    {float} *input,
    {float} *param,
    {float} *drift,
    {float} *diffs,
    {float} *obsrv
)
{{
    {model_name}(nnode, state, input, param, drift, diffs, obsrv);
{loops}
}}
"""

    chunk_template = """
unsigned int idx = i * {width} * {nsvar} + j * {width} + l;
state[idx] += dt * drift[idx];
"""

    def __init__(self, modelcg):
        self.modelcg = modelcg

    @property
    def kernel_name(self):
        return 'tvb_Euler_%s' % (
            self.modelcg.model.__class__.__name__, )

    def generate_c(self, spec):
        model_code = self.modelcg.generate_code(spec, Storage.static)
        model_name = self.modelcg.kernel_name
        nsvar = self.modelcg.model.state_sym.size
        chunk = Loop('l', spec.width,
                     self.chunk_template.format(nsvar=nsvar, **spec.dict))
        inner = Loop('j', nsvar, chunk)
        outer = Loop('i', 'nnode/%d' % spec.width, inner)
        if spec.openmp:
            chunk.pragma = '#pragma omp simd'
            outer.pragma = '#pragma omp parallel for'
        return self.template.format(
            name=self.kernel_name,
            model_code=model_code,
            model_name=model_name,
            loops=indent(outer.generate_c(spec)),
            **spec.dict
        )
