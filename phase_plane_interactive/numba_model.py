import numpy
from numba import cuda, float32, guvectorize, float64


def make_model():
    "Construct CUDA device function for the model."

    # parameters
    %for c,val in const.items():
    ${c}=float32(${val})
    %endfor

    @cuda.jit(device=True)
    def f(dX, X, ${cvar}):
        t = cuda.threadIdx.x
        % for dX in dfuns:
        ${dX}
        %endfor


    return f
