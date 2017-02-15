
"""
Handle code generation tasks. Likely to become a module with
task specific modules.

"""

template = """
#include <math.h>

void {name}(
    unsigned int nnode,
    {float} * __restrict state,
    {float} * __restrict input,
    {float} * __restrict param,
    {float} * __restrict drift,
    {float} * __restrict diffs,
    {float} * __restrict obsrv
)
{{
  {decls}
  {loop_pragma}
  for (unsigned int j=0; j < nnode; j++)
  {{
    unsigned int i = j / {width};
    {body}
  }}
}}
"""

def generate_alignments(names, spec):
    value = spec['align']
    lines = []
    for name in names:
        fmt = '{name} = __builtin_assume_aligned({name}, {value});'
        line = fmt.format(name=name, value=value)
        lines.append(line)
    return lines

def generate_code(model, spec):
    decls = generate_alignments(
        'state input param drift diffs obsrv'.split(), spec)
    decls += model.declarations(spec)
    body = model.inner_loop_lines(spec)
    code = template.format(
        decls='\n  '.join(decls),
        body='\n    '.join(body),
        name=model.kernel_name,
        nsvar=len(model.state_sym),
        loop_pragma='#pragma omp simd safelen(%d)' % (spec['width'], ),
        **spec
    )
    if spec['float'] == 'float':
        code = code.replace('pow', 'powf')
    return code

