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

/** \file interp.hpp
 * A brief description.
 * A longer description.
 */
 
#ifndef TVB_interp
#define TVB_interp

#include <vector>
#include <cmath>

namespace tvb {

    template <typename container_type, typename query_type> class interp_zero/*{{{*/
    {
    public:
        using value_type = typename container_type::value_type;
        using query_type = typename query_type;
        interp_zero(query_type step) : _step(step) { }
        value_type operator()(container_type container, query_type query) { return container[query / _step]; }
    private:
        const query_type _step;
    };/*}}}*/

    template <typename container_type, typename query_type> class interp_round/*{{{*/
    {
    public:
        using value_type = typename container_type::value_type;
        using query_type = typename query_type;
        interp_zero(query_type step) : _step(step) { }
        value_type operator()(container_type container, query_type query) { return container[round(query / _step)]; }
    private:
        const query_type _step;
    };/*}}}*/

    template <typename container_type, typename query_type> class interp_linear/*{{{*/
    {
    public:
        using value_type = typename container_type::value_type;
        using query_type = typename query_type;
        interp_zero(query_type step) : _step(step) { }
        value_type operator()(container_type& container, query_type query)
        {
            size_t index  = query / _step;
            value_type left = container[index];
            value_type right = container[index + 1];
            value_type slope = (right - left) / _step;
            value_type dquery = query % _step;
            return slope * dquery + left;
        }
    private:
        const query_type _step;
    };/*}}}*/

}; // namespace tvb
#endif // TVB_interp
