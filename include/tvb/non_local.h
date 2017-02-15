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

/** \file non_local.hpp
 * A brief description.
 * A longer description.
 */
 
#ifndef TVB_non_local
#define TVB_non_local

#include <vector>
#include <cmath>

#include "tvb/util.h"
#include "tvb/coupling.h"
#include "tvb/connectome.h"

namespace tvb {

    /** non-local / long-range projection model w/ time delays
     *
     */
    template <typename _coupling_type>
    class non_local
    {
    public:
        using coupling_type = _coupling_type;

        // if value_type is a chunk, we amortize lookup over par sweep
        // points. but have so far assumed value_type has arithmetic.
        using value_type = typename coupling_type::value_type;
        using connectome_type = tvb::connectome<value_type>;

        non_local(value_type time_step,
                value_type speed,
                size_t decim,
                coupling_type& coupling,
                connectome_type& connectome
                )
            : _coupling(coupling)
              , _connectome(connectome)
              , _time_step(time_step)
              , _recip_speed_step(1.0 / speed / time_step / decim)
              , _decim(decim)
              , _speed(speed)
        {
            buf.resize(n_node());
            buf_pos.assign(n_node(), 0);
            for (size_t i = 0; i < n_node(); i++)
            {
                size_t horizon = connectome.max_length(i) / _speed / _time_step / decim;
                buf.at(i).resize(horizon + 2);
            }
        }

        size_t n_node() { return _connectome.n_node(); }

        // unify eff and aff as a single in/out vec<chunk>
        void step(std::vector<value_type> efferent,
                  std::vector<value_type> afferent)
        {
            if (_step_count % _decim)
            {
                _step_count += 1;
                return;
            }
            for (size_t i = 0; i < n_node(); i++)
            {
                _update_node(i, efferent[i]);
                afferent[i] = _query_node(i, efferent[i]);
            }
        }

    private:
        value_type _get_delayed(size_t i, size_t j)
        {
            value_type delay = _connectome.length(i, j) * _recip_speed_step;
            size_t buf_idx = buf_pos[j] - round(delay);
            return buf[j][wrap(buf_idx, buf[j].size())];
        }

        void _update_node(size_t i, value_type current)
        {
            size_t pos = wrap(buf_pos[i] + 1, buf[i].size());
            buf[i][pos] = current;
            buf_pos[i] = pos;
        }

        value_type _query_node(size_t i, value_type current)
        {
            value_type acc = 0;
            for (size_t j = 0; j < n_node(); j++)
            {
                value_type w_ij = _connectome.weight(i, j);
                if (w_ij != 0)
                {
                    value_type post_syn = _get_delayed(i, j);
                    value_type pre_syn = current;
                    acc += _coupling.pre_sum(pre_syn, post_syn);
                }
            }
            return _coupling.post_sum(acc);
        }

        const value_type _speed;
        const value_type _time_step;
        const value_type _recip_speed_step;
        const size_t _decim;
        size_t _step_count;
        coupling_type& _coupling;
        connectome<value_type>& _connectome;
        std::vector<size_t> buf_pos;
        std::vector<std::vector<value_type> > buf;

    };

}; // namespace tvb
#endif // TVB_non_local
