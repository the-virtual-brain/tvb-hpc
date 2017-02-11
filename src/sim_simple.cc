//     Copyright 2017 TVB-HPC contributors
// 
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
// 
//        http://www.apache.org/licenses/LICENSE-2.0
// 
//     Unless required by applicable law or agreed to in writing, software
//     distributed under the License is distributed on an "AS IS" BASIS,
//     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//     See the License for the specific language governing permissions and
//     limitations under the License.

/** \file sim_simple.cc
 *
 * A simple walkthrough of API in building simple simulator
 */

#include <vector>
#include "tvb/non_local.h"
#include "tvb/rww.h"
#include "tvb/euler.h"

// Use C linkage and prefixed name so trivially callable from Python ctypes.

extern "C" {

void tvb_sim_simple()
{
    using model_t = tvb::rww<8, float>;
    using scheme_t = tvb::euler<model_t>;
    using coupl_t = tvb::linear_coupling<float>;
    using nl_t = tvb::non_local<coupl_t>;
    using conn_t = tvb::connectome<float>;

    model_t model;

    scheme_t scheme;
    scheme.dt() = 0.1;

    coupl_t coupl(1e-2, 0.0);
    conn_t conn(128);
    nl_t nl(scheme.dt(), 1.0, 5, coupl, conn);

    // no API yet for below
    std::vector<model_t::state_type> state(128 / 8);
    std::vector<model_t::coupling_type> coupling(128 / 8);

    std::vector<model_t::value_type> eff(128), aff(128);

    size_t iter = 0;
    for (;;)
    {
        nl.step(eff, aff);

        // rw nl for zero copy?
        for (int i=0; i<(128/8); i++)
            for (int j=0; j<8; j++)
                coupling[i](0, j) = aff[i*8 + j];

        for (int i=0; i<(128/8); i++)
            scheme.eval(state[i], coupling[i], model);

        // rw nl for zero copy?
        for (int i=0; i<(128/8); i++)
            for (int j=0; j<8; j++)
                 eff[i*8 + j] = state[i](0, j);

        // TODO monitors

        iter += 1;

        if (iter > 100)
            break;
    }
}
}
