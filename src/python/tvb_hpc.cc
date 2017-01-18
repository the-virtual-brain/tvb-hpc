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

#include "pybind11/pybind11.h"
#include "tvb/util.h"

namespace tvb {
namespace py {

    namespace py = pybind11;

    void add_functions(py::module& m)
    {
        m.def("wrap", &tvb::wrap<int>, "wrap integer");
    }

    void add_classes(py::module& m)
    {

    }

    void add_variables(py::module& m)
    {
    }

    auto build_module()
    {
        py::module m("tvb_hpc", "TVB for HPC");
        add_functions(m);
        add_classes(m);
        return m.ptr();
    }

}; // namespace py
}; // namespace tvb

PYBIND11_PLUGIN(tvb_hpc) { return tvb::py::build_module(); }
