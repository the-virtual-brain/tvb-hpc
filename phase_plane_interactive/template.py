class ${name}:

    def __init__(self,\
%for c,val in const.items():
${c}=${val}${'' if loop.last else ', '}\
%endfor
):
        self.limit = ${limit}
        % for c in const.keys():
        self.${c} = ${c}
        % endfor

    def dfun(self,state_variables, *args, **kwargs):
        <% 
        sv_csl = ", ".join(sv) 
        %> 
        ${sv_csl} = state_variables

        % for c in const.keys():
        ${c} = self.${c}
        % endfor

        % for cvar in input:
        ${cvar} = 0.0
        % endfor
        
        % for i, var in enumerate(sv):
        d${var} = ${drift[i]}
        % endfor
        return [${sv_csl}] 


