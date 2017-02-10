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

/** \file util.hpp
 * A brief description.
 * A longer description.
 */
 
#ifndef TVB_util
#define TVB_util

#include <vector>

namespace tvb {

    /** wraps integer values within range.
     *
     */
    template <typename I> I wrap(I idx, I len)
    {
        I idx_ = idx % len + (idx < 0) * len;
        return idx_ * (idx_ != len);
    }
    
    /** wrapper for SIMD-friendly chunk of data
     *
     */
    template <size_t _chunk_size, size_t _n_svar, typename _value_type=float>
    class chunk
    {
    public:
        using value_type = _value_type;

        chunk(value_type *data) : _data(data) { }

        chunk() : _vector(_chunk_size * _n_svar) {
            _data = _vector.data();
        }

        static size_t width() { return _chunk_size; }
        static size_t length() { return _n_svar; }

        value_type& operator()(size_t var_idx, size_t lane_idx) {
            return _data[var_idx*_chunk_size + lane_idx];
        }

        void fill_var(size_t var_idx, value_type value) {
            for (size_t lane_idx=0; lane_idx<_chunk_size; lane_idx++)
                (*this)(var_idx, lane_idx) = value;
        }

        void fill_lane(size_t lane_idx, value_type value) {
            for (size_t var_idx=0; var_idx<_chunk_size; var_idx++)
                (*this)(var_idx, lane_idx) = value;
        }

        void fill(value_type value) {
            for (size_t var_idx=0; var_idx<_n_svar; var_idx++)
                fill_var(var_idx, value);
        }

    private:
        value_type *_data;
        std::vector<value_type> _vector;
    };

}; // namespace tvb
#endif // TVB_util
