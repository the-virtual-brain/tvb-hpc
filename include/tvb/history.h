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
#include <cmath>

namespace tvb {

    template <typename I> I wrap(I idx, I len) { return idx % len + (idx < 0) * len; };

    template <typename value_type> class ring_buffer
    {
    public:
        using value_type = typename value_type;
        DRing(size_t size) : buf(size) { }
        template <typename I> T& operator[](const I idx) { return buf[wrap(idx, buf.size())]; }
        size_t size() { return buf.size(); }
    private:
        std::vector<value_type> buf;
    };

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

    void test()
    {
        using rb_t = ring_buffer<float>;
        using interp_t = interp_linear<rb, float>;
        
        rb_t rb(32);
        interp_t interp(0.1f);
        interp(rb, 234.024);
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
