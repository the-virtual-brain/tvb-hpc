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

/** \file rww.h
 * Reduced Wong-Wang model
 */
 
#ifndef TVB_rww
#define TVB_rww

#include <array>
#include <cmath>
#include "tvb/util.h"

namespace tvb {

    template <size_t _chunk_size=4, typename _value_type=float>
    class rww
    {

    public:
        using value_type = _value_type;
        using chunk_type = tvb::chunk<_chunk_size, 1, value_type>;

        rww() {
            // default values as provided in TVB
            _a = 0.270;
            _b = 0.108;
            _d = 154.0;
            _g = 0.641;
            _ts = 100.0;
            _w = 0.6;
            _j = 0.2609;
            _io = 0.33;
        }

        void eval(chunk_type state,
                  chunk_type deriv,
                  chunk_type coupling
                  )
        {
            value_type S, c, x, h, dS, above_one, below_zero;
            for (size_t i=0; i<_chunk_size; i++)
            {
                S = state.at(0, i);
                c = coupling.at(0, i);
                x = w()*j()*S + io() + j()*c;
                h = (a()*x - b()) / (1 - exp(-d()*(a()*x - b())));
                dS = - (S / ts()) + (1 - S) * h * g();
                deriv.at(0, i) = (1 - S) * (S > 1) + (0 - S) * (S < 0) + (S > 0) * (S < 1) * dS;
            }
        }

        value_type& a() { return _a; }
        value_type& w() { return _w; }
        value_type& j() { return _j; }
        value_type& io() { return _io; }
        value_type& b() { return _b; }
        value_type& d() { return _d; }
        value_type& ts() { return _ts; }
        value_type& g() { return _g; }

    private:
        value_type _a, _w, _j, _io, _b, _d, _g, _ts;

    };

}; // namespace tvb
#endif // TVB_rww
