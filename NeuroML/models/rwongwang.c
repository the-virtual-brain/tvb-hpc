#include <stdio.h> // for printf
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

#include <curand_kernel.h>
#include <curand.h>
    
__device__ float wrap_it_V(float V)
{
    int Vdim[] = {-2.0, 4.0};
    if (V < V[0]) return V[0];
    else if (V > V[1]) return V[1];
}
__device__ float wrap_it_W(float W)
{
    int Wdim[] = {-6.0, 6.0};
    if (W < W[0]) return W[0];
    else if (W > W[1]) return W[1];
}

    __shared__ const float w_plus = 1.4f
    __shared__ const float a_E = 310.0f
    __shared__ const float b_E = 125.0f
    __shared__ const float d_E = 0.154f
    __shared__ const float a_I = 615.0f
    __shared__ const float b_I = 177.0f
    __shared__ const float d_I = 0.087f
    __shared__ const float gamma_E = 0.641f / 1000.0f
    __shared__ const float tau_E = 100.0f
    __shared__ const float tau_I = =10.0f
    __shared__ const float I_0 = 0.382f
    __shared__ const float w_E = 1.0f
    __shared__ const float w_I = 0.7f
    __shared__ const float gamma_I = 1.0f / 1000.0f
    __shared__ const float min_d_E = -1.0f * d_E
    __shared__ const float min_d_I = -1.0f * d_I
    __shared__ const float imintau_E = -1.0f / tau_E
    __shared__ const float imintau_I = -1.0f / tau_I
    __shared__ const float w_E__I_0 = w_E * I_0
    __shared__ const float w_I__I_0 = w_I * I_0
    __shared__ const float J_N = 0.15
    __shared__ const float J_I = 1.0
    __shared__ const float G = 2.0
    __shared__ const float lamda = 0.0

__global__ void rWongWang(

        // config
        unsigned int i_step, unsigned int n_node, unsigned int nh, unsigned int n_step, unsigned int n_params,
        float dt, float speed, float * __restrict__ weights, float * __restrict__ lengths,
        float * __restrict__ params_pwi, // pwi: per work item
        // state
        float * __restrict__ state_pwi,
        // outputs
        float * __restrict__ tavg_pwi
        )
{
    // work id & size
    const unsigned int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned int size = blockDim.x * gridDim.x * gridDim.y;

#define params(i_par) (params_pwi[(size * (i_par)) + id])
// TODO: the scaler for state is the number of statevariables?
#define state(time, i_node) (state_pwi[((time) * 2 * n_node + (i_node))*size + id])
#define tavg(i_node) (tavg_pwi[((i_node) * size) + id])

    // unpack params
    // These are the two parameters which are usually explore in fitting in this model
    const float global_coupling = params(0);
    const float global_speed = params(1);

    const float nsig = sqrt(dt) * sqrt(2.0 * 1e-3);
    const float rec_speed_dt = 1.0f / global_speed / (dt);
    const float lc_0 = 0.0;

    curandState s;
    curand_init(id * (blockDim.x * gridDim.x * gridDim.y), 0, 0, &s);

    double tmp_I_E = 0.0;
    double tmp_H_E = 0.0;
    double tmp_I_I = 0.0;
    double tmp_H_I = 0.0;
    double dV = 0.0;
    double dW = 0.0;
    double V = 0.0;
    double W = 0.0;

    //***// This is only initialization of the observable
    for (unsigned int i_node = 0; i_node < n_node; i_node++)
        tavg(i_node) = 0.0f;

    //***// This is the loop over time, should stay always the same
    for (unsigned int t = i_step; t < (i_step + n_step); t++)
    {
    //***// This is the loop over nodes, which also should stay the same
        for (unsigned int i_node = threadIdx.y; i_node < n_node; i_node+=blockDim.y)
        {
            c_0 = 0.0f;
        // is this a correct indexing for n_node?
            V = state((t) % nh, i_node + 0 * n_node);
            W = state((t) % nh, i_node + 1 * n_node);
            for (unsigned int j_node = 0; j_node < n_node; j_node++)
            {
                //***// Get the weight of the coupling between node i and node j
                float wij = weights[i_n + j_node]; // nb. not coalesced
                if (wij == 0.0)
                    continue;
                //***// Get the delay between node i and node j
                unsigned int dij = lengths[i_n + j_node] * rec_speed_dt;
                //***// Get the state of node j which is delayed by dij
                float V_j = state((t - dij + NH) % NH, j_node);
                //***// Sum it all together using the coupling function. This is a kuramoto coupling so: a * sin(pre_syn - post_syn)
                c_0 += wij * sin(V_j - V);
            } // j_node */

            // rec_n is only used for the scaling over nodes for kuramoto, for python this scaling is included in the post_syn
            c_0 = global_coupling * global_speed;

            // This is dynamics step and the update in the state of the node
            tmp_I_E = a_E * (w_E__I_0 + w_plus__J_NMDA * S_E + sum - JI*S_I) - b_E;
            tmp_H_E = tmp_I_E/(1.0-exp(min_d_E * tmp_I_E));
            tmp_I_I = (a_I*((w_I__I_0+(J_NMDA * S_E))-S_I))-b_I;
            tmp_H_I = tmp_I_I/(1.0-exp(min_d_I*tmp_I_I));
            dV = (imintau_E* S_E)+(tmp_H_E*(1-S_E)*gamma_E);
            dW = (imintau_I* S_I)+(tmp_H_I*gamma_I);

            //***// Add noise (if noise components are present in model), integrate with stochastic forward euler and wrap it up
            V += dt * (sig * curand_normal(&s) + tmp_I_E);
            W += dt * (sig * curand_normal(&s) + tmp_H_E);

            // Wrap it within the limits of the model
            wrap_it_V(V);
            wrap_it_W(W);

            // Update the state
            state((t + 1) % nh, i_node + 0 * n_node) = V;
            state((t + 1) % nh, i_node + 1 * n_node) = W;

            // Update the observable
            tavg(i_node + 0 * n_node) = sin(V);
            tavg(i_node + 1 * n_node) = sin(W);

            // sync across warps executing nodes for single sim, before going on to next time step
            __syncthreads();

        } // for i_node
    } // for t

// cleanup macros/*{{{*/
#undef params
#undef state
#undef tavg/*}}}*/

} // kernel integrate