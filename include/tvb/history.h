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

/** \file history.hpp
 * A brief description.
 * A longer description.
 */
 
#ifndef TVB_history
#define TVB_history

#include <vector>

namespace tvb {

    template <typename T> class ring_buffer
    {
    public:
        DRing(size_t size) buf(size) { }
    private:
        std::vector<T> buf;
    }

    template <typename Tx=float, typename Ty=Tx> class CRing // {{{
    {
    public:
        CRing(Tx length, Tx step) : _origin(0), _length(length), _step(step), _buf((length / step) + 2) { }
        Tx origin() { return _origin; }
        Tx length() { return _length; }
        Tx step() { return _step; }
        void move(Tx dx) { _origin += dx; };
        Ty & at(Tx x) { return buf_at((_origin + x) / _step); }

    private:
        Tx _origin, _length, _step;
        std::vector<Ty> _buf;

        Ty & buf_at(const int idx_)
        {
            int idx = idx_;

            if (idx < 0)
                idx = _buf.size() - ( (-idx) % _buf.size());

            else if (idx > _buf.size())
                idx %= _buf.size();

            return _buf[idx];
        }
    }; // }}}

}; // namespace tvb
#endif // TVB_history
