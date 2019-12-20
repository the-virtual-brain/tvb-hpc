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
#include <stdbool.h>

    
__device__ float wrap_it_x1(float x1)
{
    int x1dim[] = {-2., 1.};
    if (x1 < x1[0]) return x1[0];
    else if (x1 > x1[1]) return x1[1];
}
__device__ float wrap_it_y1(float y1)
{
    int y1dim[] = {-20., 2.};
    if (y1 < y1[0]) return y1[0];
    else if (y1 > y1[1]) return y1[1];
}
__device__ float wrap_it_z(float z)
{
    int zdim[] = {-2.0, 5.0};
    if (z < z[0]) return z[0];
    else if (z > z[1]) return z[1];
}
__device__ float wrap_it_x2(float x2)
{
    int x2dim[] = {-2., 0.};
    if (x2 < x2[0]) return x2[0];
    else if (x2 > x2[1]) return x2[1];
}
__device__ float wrap_it_y2(float y2)
{
    int y2dim[] = {0., 2.};
    if (y2 < y2[0]) return y2[0];
    else if (y2 > y2[1]) return y2[1];
}
__device__ float wrap_it_g(float g)
{
    int gdim[] = {-1, 1.};
    if (g < g[0]) return g[0];
    else if (g > g[1]) return g[1];
}

    __shared__ const float a = 1.0
    __shared__ const float b = 3.0
    __shared__ const float c = 1.0
    __shared__ const float d = 5.0
    __shared__ const float r = 0.00035
    __shared__ const float s = 4.0
    __shared__ const float x0 = -1.6
    __shared__ const float Iext = 3.1
    __shared__ const float slope = 0.
    __shared__ const float Iext2 = 3.1
    __shared__ const float tau = 10.0
    __shared__ const float aa = 6.0
    __shared__ const float bb = 2.0
    __shared__ const float Kvf = 0.0
    __shared__ const float Kf = 0.0
    __shared__ const float Ks = 0.0
    __shared__ const float tt = 1.0
    __shared__ const float modification = 1.0

    // coupling contants
    __shared__ coupl_a = 1
    __shared__ coupl_b = 0.1

__global__ void Epileptor_2D(

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
#define state(time, i_node) (state_pwi[((time) * 6 * n_node + (i_node))*size + id])
#define tavg(i_node) (tavg_pwi[((i_node) * size) + id])

    // unpack params
    // These are the two parameters which are usually explore in fitting in this model
    const float global_coupling = params(0);
    const float global_speed = params(1);
    const float x0 = params(2);

    const float sig_x1 = sqrt(dt) * sqrt(2.0 * 1e-3);
    const float sig_y1 = sqrt(dt) * sqrt(2.0 * 1e-3);
    const float sig_z = sqrt(dt) * sqrt(2.0 * 1e-3);
    const float sig_x2 = sqrt(dt) * sqrt(2.0 * 1e-3);
    const float sig_y2 = sqrt(dt) * sqrt(2.0 * 1e-3);
    const float sig_g = 0;
    const float rec_speed_dt = 1.0f / global_speed / (dt);
    const float lc_0 = 0.0;

    curandState s;
    curand_init(id * (blockDim.x * gridDim.x * gridDim.y), 0, 0, &s);

    double dx1 = 0.0;
    double dy1 = 0.0;
    double dz = 0.0;
    double dx2 = 0.0;
    double dy2 = 0.0;
    double dg = 0.0;
    double x1 = 0.0;
    double y1 = 0.0;
    double z = 0.0;
    double x2 = 0.0;
    double y2 = 0.0;
    double g = 0.0;

    //***// This is only initialization of the observable
    for (unsigned int i_node = 0; i_node < n_node; i_node++)
        tavg(i_node) = 0.0f;

    //***// This is the loop over time, should stay always the same
    for (unsigned int t = i_step; t < (i_step + n_step); t++)
    {
    //***// This is the loop over nodes, which also should stay the same
        for (unsigned int i_node = threadIdx.y; i_node < n_node; i_node+=blockDim.y)
        {
            c_pop1 = 0.0f;
            c_pop2 = 0.0f;

            x1 = state((t) % nh, i_node + 0 * n_node);
            y1 = state((t) % nh, i_node + 1 * n_node);
            z = state((t) % nh, i_node + 2 * n_node);
            x2 = state((t) % nh, i_node + 3 * n_node);
            y2 = state((t) % nh, i_node + 4 * n_node);
            g = state((t) % nh, i_node + 5 * n_node);

            // This variable is used to traverse the weights and lengths matrix, which is really just a vector. It is just a displacement.
            unsigned int i_n = i_node * n_node;

            for (unsigned int j_node = 0; j_node < n_node; j_node++)
            {
                //***// Get the weight of the coupling between node i and node j
                float a = weights[i_n + j_node]; // nb. not coalesced
                if (a == 0.0)
                    continue;

                //***// Get the delay between node i and node j
                unsigned int dij = lengths[i_n + j_node] * rec_speed_dt;

                //***// Get the state of node j which is delayed by dij
                float x1_j = state((t - dij + nh) % nh), j_node + 0 * n_node);
                float y1_j = state((t - dij + nh) % nh), j_node + 1 * n_node);

                // Sum it all together using the coupling function. This is a kuramoto coupling so: (postsyn * presyn) == ((a) * (sin(xj - xi))) 
                c_pop1 = wij * a * sin(x1_j - x1)

            } // j_node */

            // rec_n is only used for the scaling over nodes for kuramoto, for python this scaling is included in the post_syn
            c_pop1 *= global_coupling
            c_pop2 *= g

            // The conditional variables
            if_ydot0 = (x1 < 0) * (- a * powf(x1, 2) * b * x1);
            else_ydot0 = (x1 >= 0) * (slope - x2 + 0.6 * powf((z - 4.0), 2));
            if_ydot1 = (y1 < 0) * (- 0.1 * powf(Z, 7));
            else_ydot1 = 0;
            ifmod_h = (modification==true)*(x0 + 3. / (1. + exp(-(x1 + 0.5) / 0.1)) ;
            elsemod_h = (modification==false)*(4 * (x1 - x0) + if_ydot1) );
            if_ydot3 = 0;
            else_ydot3 = (x2 >= -0.25) * (aa * (x2 + 0.25));

            // This is dynamics step and the update in the state of the node
            dx1 = tt * (y1 - z + Iext + Kvf * c_pop1 + (if_ydot0 + else_ydot0) );
            dy1 = tt * (c - d * powf(x1, 2) - y1);
            dz = tt * (r * ((ifmod_h + elsemod_h)) - z + Ks * c_pop1));
            dx2 = tt * (-y2 + x2 - powf(x2, 3) + Iext2 + bb * g - 0.3 * (z - 3.5) + Kf * c_pop2);
            dy2 = tt * (-y2 + else_ydot3) / tau;
            dg = tt * (-0.01 * (g - 0.1 * x1) );

            // Add noise (if noise components are present in model), integrate with stochastic forward euler and wrap it up
            x1 += dt * (sig_x1 * curand_normal(&s) + dx1);
            y1 += dt * (sig_y1 * curand_normal(&s) + dy1);
            z += dt * (sig_z * curand_normal(&s) + dz);
            x2 += dt * (sig_x2 * curand_normal(&s) + dx2);
            y2 += dt * (sig_y2 * curand_normal(&s) + dy2);
            g += dt * (sig_g * curand_normal(&s) + dg);

            // Wrap it within the limits of the model
            wrap_it_x1(x1);
            wrap_it_y1(y1);
            wrap_it_z(z);
            wrap_it_x2(x2);
            wrap_it_y2(y2);
            wrap_it_g(g);

            // Update the state
            state((t + 1) % nh, i_node + 0 * n_node) = x1;
            state((t + 1) % nh, i_node + 1 * n_node) = y1;
            state((t + 1) % nh, i_node + 2 * n_node) = z;
            state((t + 1) % nh, i_node + 3 * n_node) = x2;
            state((t + 1) % nh, i_node + 4 * n_node) = y2;
            state((t + 1) % nh, i_node + 5 * n_node) = g;

            // Update the observable only for the last timestep
            if (t == (i_step + n_step - 1)){
                tavg(i_node + 0 * n_node) = x1;
                tavg(i_node + 1 * n_node) = x2;
                tavg(i_node + 2 * n_node) = z;
                tavg(i_node + 3 * n_node) = -x1 + x2;
            }

            // sync across warps executing nodes for single sim, before going on to next time step
            __syncthreads();

        } // for i_node
    } // for t

// cleanup macros/*{{{*/
#undef params
#undef state
#undef tavg/*}}}*/

} // kernel integrate