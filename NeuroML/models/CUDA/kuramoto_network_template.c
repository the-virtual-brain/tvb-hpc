#define PI_2 (2 * M_PI_F)

// buffer length defaults to the argument to the integrate kernel
// but if it's known at compile time, it can be provided which allows
// compiler to change i%n to i&(n-1) if n is a power of two.
#ifndef NH
#define NH nh
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#include <stdio.h> // for printf
#include <curand_kernel.h>
#include <curand.h>

__device__ float wrap_2_pi_(float x)/*{{{*/
{
    bool neg_mask = x < 0.0f;
    bool pos_mask = !neg_mask;
    // fmodf diverges 51% of time
    float pos_val = fmodf(x, PI_2);
    float neg_val = PI_2 - fmodf(-x, PI_2);
    return neg_mask * neg_val + pos_mask * pos_val;
}/*}}}*/

__device__ float wrap_2_pi(float x) // not divergent/*{{{*/
{
    bool lt_0 = x < 0.0f;
    bool gt_2pi = x > PI_2;
    return (x + PI_2)*lt_0 + x*(!lt_0)*(!gt_2pi) + (x - PI_2)*gt_2pi;
}/*}}}*/

__global__ void integrate(/*{{{*/
        // config
        unsigned int i_step, unsigned int n_node, unsigned int nh, unsigned int n_step, unsigned int n_params,
        float dt, float speed,
        float * __restrict__ weights,
        float * __restrict__ lengths,
        float * __restrict__ params_pwi, // pwi: per work item
        // state
        float * __restrict__ state_pwi,
        // outputs
        float * __restrict__ tavg_pwi
        )
{/*}}}*/

    // work id & size/*{{{*/
    const unsigned int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned int size = blockDim.x * gridDim.x * gridDim.y;/*}}}*/

    // ND array accessors (TODO autogen from py shape info)/*{{{*/
#define params(i_par) (params_pwi[(size * (i_par)) + id])
#define state(time, i_node) (state_pwi[((time) * n_node + (i_node))*size + id])
#define tavg(i_node) (tavg_pwi[((i_node) * size) + id])/*}}}*/

    // unpack params/*{{{*/
    //***// These are the two parameters which are usually explore in fitting in this model
    const <TYPE_1> <PARAMETER_1> = params(1);
    const <TYPE_2> <PARAMETER_2> = params(0);/*}}}*/

    // derived/*{{{*/
    const float <RECN_NAME> = <RECN_VALUE>;
    //***// The speed affects the delay and speed_value is a parameter which is usually explored in fitting ***
    const float <RECSPEED_NAME> = <RECSPEED_VALUE>;
    //***// This is a parameter specific to the Kuramoto model
    const float <OMEGA_NAME> = <OMEGA_VALUE>;
    //***// This is a parameter for the stochastic integration step, you can leave this constant for the moment
    const float <NOISE_NAME> = <NOISE_VALUE>;/*}}}*/ //-->noise sigma value

    curandState s;
    curand_init(id * (blockDim.x * gridDim.x * gridDim.y), 0, 0, &s);

    //***// This is only initialization of the observable
    for (unsigned int i_node = 0; i_node < n_node; i_node++)
        tavg(i_node) = 0.0f;

    //***// This is the loop over time, should stay always the same
    for (unsigned int t = i_step; t < (i_step + n_step); t++)
    {
    //***// This is the loop over nodes, which also should stay the same
        for (unsigned int i_node = threadIdx.y; i_node < n_node; i_node+=blockDim.y)
        {
            //***// We here gather the current state of the node
            float theta_i = state(t % NH, i_node);
            //***// This variable is used to traverse the weights and lengths matrix, which is really just a vector. It is just a displacement.
            unsigned int i_n = i_node * n_node;
            float sum = 0.0f;

           //***// For all nodes that are not the current node (i_node) sum the coupling
            for (unsigned int j_node = 0; j_node < n_node; j_node++)
            {
            //***// Get the weight of the coupling between node i and node j
                float wij = weights[i_n + j_node]; // nb. not coalesced
                if (wij == 0.0)
                    continue;
                //***// Get the delay between node i and node j
                unsigned int dij = lengths[i_n + j_node] * rec_speed_dt;
                //***// Get the state of node j which is delayed by dij
                float theta_j = state((t - dij + NH) % NH, j_node);
                //***// Sum it all together using the coupling function. This is a kuramoto coupling so: a * sin(pre_syn - post_syn)
                <COUPLING_FUNCTION_NAME> <COUPLING_FUNCTION_REDUCE> <COUPLING_FUNCTION_VALUE>;
            } // j_node

            //***// This is actually the integration step and the update in the state of the node
            <MODEL_STATE> += <INTEGRATION_SCHEME> * <MODEL_DYNAMICS>;
            //***// We add some noise if noise is selected
            <MODEL_STATE> += <NOISE_PARAM> * curand_normal2(&s).x;
            //***// Wrap it within the limits of the model (0-2pi)
            <MODEL_STATE> = wrap_<MODEL_LIMIT>_(<MODEL_STATE>);
            //***// Update the state
            state((t + 1) % NH, i_node) = <MODEL_STATE>;
            //***// Update the observable
            tavg(i_node) = <OBSERV_OUTPUT>;

            // sync across warps executing nodes for single sim, before going on to next time step
            __syncthreads();

        } // for i_node
    } // for t

// cleanup macros/*{{{*/
#undef params
#undef state
#undef tavg/*}}}*/

} // kernel integrate