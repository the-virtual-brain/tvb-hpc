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

 
#ifndef TVB_euler
#define TVB_euler

namespace tvb {

    /** Euler time stepping integration scheme
     *
     */
    template <typename _model_type> 
    class euler {
    public:
        using model_type = _model_type;
        using value_type = typename _model_type::value_type;
        using state_type = typename _model_type::state_type;
        using coupling_type = typename _model_type::coupling_type;
        
        euler() { }

        void eval(state_type state, coupling_type coupling, model_type model)
        {
            model.eval(state, _deriv, coupling);

            for (size_t i=0; i<state_type::length(); i++)
                for (size_t j=0; j<state_type::width(); j++)
                    state.at(i, j) += _dt * _deriv.at(i, j);
        }

        value_type& dt() { return _dt; }

    private:
        value_type _dt;
        typename model_type::state_type _deriv;
    };
}; // namespace tvb
#endif // TVB_euler
