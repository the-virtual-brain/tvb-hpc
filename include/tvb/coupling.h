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

/** \file coupling.hpp
 * A brief description.
 * A longer description.
 */
 
#ifndef TVB_coupling
#define TVB_coupling

#include <vector>
#include <cmath>

namespace tvb {

    /** example linear coupling
     *
     */
    template <typename value_type=float> class linear_coupling
    {
    public:
        linear_coupling(value_type slope, value_type offset)
            : _slope(slope), _offset(offset) { }
        value_type pre_sum(value_type pre_syn, value_type post_syn) { return pre; }
        value_type post_sum(value_type sum) { return _slope * sum + _offset; }
    private:
        const value_type _slope, _offset;
    }

}; // namespace tvb
#endif // TVB_coupling
