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

#ifndef TVB_hmje
#define TVB_hmje

#include <array>
#include <cmath>
#include "tvb/util.h"

namespace tvb {

    /** \class hmje.h
     * Hindmarsh-Rose-Jirsa Epileptor
     */
    template <size_t _chunk_size=4, typename _value_type=float>
    class hmje
    {

    public:
        using value_type = _value_type;
        using state_type = tvb::chunk<_chunk_size, 6, value_type>;
        using coupling_type = tvb::chunk<_chunk_size, 2, value_type>;

        hmje() {
            // default values as provided in TVB
            _x0 = -1.6;
            _Iext = 3.1;
            _Iext2 = 0.45;
            _a = 1;
            _b = 3;
            _slope = 0;
            _tt = 1;
            _Kvf = 0.0;
            _c = 1;
            _d = 5;
            _r = 0.00035;
            _Ks = 0.0;
            _Kf = 0.0;
            _aa = 6;
            _tau = 10;
        }

        void eval(state_type state,
                  state_type deriv,
                  coupling_type coupling
                  )
        {
            value_type x1, y1, z, x2, y2, g, c1, c2;
            value_type dx1, dy1, dz, dx2, dy2, dg;
            value_type if_x1_pos, if_x1_neg, if_z_neg, if_x2_gt;

            for (size_t i=0; i<_chunk_size; i++)
            {
                x1 = state(0, i);
                y1 = state(1, i);
                z = state(2, i);
                x2 = state(3, i);
                y2 = state(4, i);
                g = state(5, i);

                c1 = coupling(0, i);
                c2 = coupling(1, i);

                // faster oscillator
                if_x1_neg = (x1 <  0) * (-a() * x1 * x1 + b() * x1);
                if_x1_pos = (x1 >= 0) * (slope() - x2 + 0.6 * (z - 4) * (z - 4));
                dx1 = tt() * (y1 - z + Iext() + Kvf() * c1
                        + (if_x1_neg + if_x1_pos) * x1);
                dy1 = tt() * (c() - d() * x1 * x1 - y1);

                // energy / slow variable
                if_z_neg = (z < 0) * (-0.1 * pow(z, 7));
                dz = tt() * (r() * ( 4 * (x1 - x0()) - z + if_z_neg + Ks() * c1));

                // slower oscillator
                dx2 = tt() * (-y2 + x2 - x2 * x2 * x2 + Iext2() + 2 * g
                        - 0.3 * (z - 3.5) + Kf() * c2);
                if_x2_gt = (x2 >= (-3.5)) * (aa() * (x2 + 0.25));
                dy2 = tt() * ((-y2 + if_x2_gt) / tau());

                // low pass filter
                dg = tt() * (-0.01 * (g - 0.1 * x1));

                deriv(0, i) = dx1;
                deriv(1, i) = dy1;
                deriv(2, i) = dz;
                deriv(3, i) = dx2;
                deriv(4, i) = dy2;
                deriv(5, i) = dg;
            }
        }

        value_type& x0() { return _x0; }
        value_type& Iext() { return _Iext; }
        value_type& Iext2() { return _Iext2; }
        value_type& a() { return _a; }
        value_type& b() { return _b; }
        value_type& slope() { return _slope; }
        value_type& tt() { return _tt; }
        value_type& Kvf() { return _Kvf; }
        value_type& c() { return _c; }
        value_type& d() { return _d; }
        value_type& r() { return _r; }
        value_type& Ks() { return _Ks; }
        value_type& Kf() { return _Kf; }
        value_type& aa() { return _aa; }
        value_type& tau() { return _tau; }

        static size_t n_param() { return 15; }

    private:
        value_type _x0, _Iext, _Iext2, _a, _b, _slope, _tt, _Kvf, _c, _d, _r, _Ks, _Kf, _aa, _tau;
        /* consider also storing parameters as a chunk; then, the choice
         * is whether to have per-lane values or not. accessors can map
         * chunks to values with internal indices.. but how would they get
         * the lane index..? seems like there's some iterator pattern lurking
         * about, but we haven't figured it out yet.
         */

    };

}; // namespace tvb
#endif // TVB_hmje
