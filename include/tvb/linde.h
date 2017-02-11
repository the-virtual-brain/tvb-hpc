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

#ifndef TVB_linde
#define TVB_linde

#include <array>
#include <cmath>
#include "tvb/util.h"

namespace tvb {

    /** Linear differential equation model
     *
     */
    template <size_t _chunk_size=4, typename _value_type=float>
    class linde
    {

    public:
        using value_type = _value_type;
        using state_type = tvb::chunk<_chunk_size, 1, value_type>;
        using coupling_type = tvb::chunk<_chunk_size, 1, value_type>;

        linde() {
            _lambda = -1;
        }

        void eval(state_type state,
                  state_type deriv,
                  coupling_type coupling
                  )
        {
            value_type x, c;
#pragma omp simd
            for (size_t i=0; i<_chunk_size; i++)
            {
                x = state(0, i);
                c = coupling(0, i);
                deriv(0, i) = lambda() * x + c;
            }
        }

        value_type& lambda() { return _lambda; }

        static size_t n_param() { return 1; }

    private:
        value_type _lambda;

    };

}; // namespace tvb
#endif // TVB_linde
