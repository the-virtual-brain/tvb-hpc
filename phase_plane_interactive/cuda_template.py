#include <stdio.h> // for printf
#define PI_2 (2 * M_PI_F)
#include <curand_kernel.h>
#include <curand.h>

__device__ float wrap_2_pi(float x)
{
    bool lt_0 = x < 0.0f;
    bool gt_2pi = x > PI_2;
    return (x + PI_2)*lt_0 + x*(!lt_0)*(!gt_2pi) + (x - PI_2)*gt_2pi;
}


__global__ void ${name}( \
unsigned int i_step, 
unsigned int n_node, 
unsigned int n_step, 
unsigned int n_params,
%for c,val in const.items():
    float ${c}=${val}${'' if loop.last else ', '}\
%endfor
){
    // Pre generated initializations
    curandState s;
    curand_init(id * (blockDim.x * gridDim.x * gridDim.y), 0, 0, &s);
    const unsigned int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned int size = blockDim.x * gridDim.x * gridDim.y;

    // ND array accessors (TODO autogen from py shape info)/*{{{*/
#define params(i_par) (params_pwi[(size * (i_par)) + id])
#define state(time, i_node) (state_pwi[((time) * n_node + (i_node))*size + id])
#define tavg(i_node) (tavg_pwi[((i_node) * size) + id])/*}}}*/


    // Initialize observers
    for (unsigned int i_node = 0; i_node < n_node; i_node++)
    {
        tavg(i_node) = 0.0f;
    }

    // Loop through time steps
    for (unsigned int t = i_step; t < (i_step + n_step); t++)
    {
	// Loop through nodes
        for (unsigned int i_node = 0; i_node < n_node; i_node++)
        {

	    // Coupling
	    float post_syn = state(t % NH, i_node);
            unsigned int i_n = i_node * n_node;
            float sum = 0.0f;

            for (unsigned int j_node = 0; j_node < n_node; j_node++)
            {
                float wij = weights[i_n + j_node];
                if (wij == 0.0)
                    continue;
                unsigned int dij = lengths[i_n + j_node] * rec_speed_dt;
                float pre_syn = state((t - dij + NH) % NH, j_node);
                sum += wij * ${pre_sum} #sin(pre_syn - post_syn);
            } 



	}
    }


       

