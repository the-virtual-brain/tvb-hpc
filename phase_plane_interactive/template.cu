void model_dfun(
        float * _dx, float *_x, float *mmpr, float*input
    )   
{       
    %for i,c in enumerate(const.keys()):
    float ${c} = mmpr[n_thr*${i}];
    %endfor
    
    %for i,v in enumerate(svar):
    float ${v} = _x[n_thr*${i}];
    %endfor

    float ${cvar} = input[0];

    %for i,dfun in enumerate(dfuns):
    _dx[n_thr*${i}] = ${dfun};
    %endfor

}

