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

/** \file connectome.hpp
 * A brief description.
 * A longer description.
 */
 
#ifndef TVB_connectome
#define TVB_connectome

#include <stddef.h>
#include <vector>
#include <cmath>

namespace tvb {

    template <typename value_type=float>
    class connectome
    {
    public:
        connectome(size_t n_node) : _n_node(n_node)
        {
            _weights.assign(n_node*n_node, 0);
            _lengths.assign(n_node*n_node, 0);
        }
        size_t n_node() { return _n_node; }

        value_type max_length(size_t idx) {
            value_type max = 0;
            for (size_t i=0; i < _n_node; i++)
            {
                value_type el = _lengths[_n_node * idx + i];
                max = el > max ? el : max;
            }
            return max;
        }

        value_type& weight(size_t i, size_t j) { return _weights[_n_node * i + j]; }

        value_type& length(size_t i, size_t j) { return _lengths[_n_node * i + j]; }

    private:
        const size_t _n_node;
        std::vector<value_type> _weights;
        std::vector<value_type> _lengths;
    };

}; // namespace tvb
#endif // TVB_connectome
