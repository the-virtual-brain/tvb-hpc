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

namespace tvb {

    /** wraps integer values within range.
     *
     */
    template <typename I> I wrap(I idx, I len) { return idx % len + (idx < 0) * len; };

}; // namespace tvb
#endif // TVB_util
