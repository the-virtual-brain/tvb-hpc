#     Copyright 2018 TVB-HPC contributors
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.


"""
Source for Balloon-Windkessel model of BOLD.

"""

from dsl_cuda.tvb_hpc import BaseModel


def _balloon_windkessel_constants():
    # defaults from Stefan 2007, cf tvb/analyzers/fmri_balloon.py
    TAU_S = 0.65
    TAU_F = 0.41
    TAU_O = 0.98
    ALPHA = 0.32
    TE = 0.04
    V0 = 4.0
    E0 = 0.4
    EPSILON = 0.5
    NU_0 = 40.3
    R_0 = 25.0
    RECIP_TAU_S = (1.0 / TAU_S)
    RECIP_TAU_F = (1.0 / TAU_F)
    RECIP_TAU_O = (1.0 / TAU_O)
    RECIP_ALPHA = (1.0 / ALPHA)
    RECIP_E0 = (1.0 / E0)
    # derived parameters
    k1 = (4.3 * NU_0 * E0 * TE)
    k2 = (EPSILON * R_0 * E0 * TE)
    k3 = (1.0 - EPSILON)

    return {
        'RECIP_TAU_S': RECIP_TAU_S,
        'RECIP_TAU_F': RECIP_TAU_F,
        'RECIP_TAU_O': RECIP_TAU_O,
        'RECIP_ALPHA': RECIP_ALPHA,
        'E0': E0,
        'RECIP_E0': RECIP_E0,
        'V0': V0,
        'k1': k1,
        'k2': k2,
        'k3': k3
    }


class BalloonWindkessel(BaseModel):
    const = _balloon_windkessel_constants()
    state = 's f v q'
    input = 'x'
    drift = (
        'x - RECIP_TAU_S * s - RECIP_TAU_F * (f - 1)',
        's',
        'RECIP_TAU_O * (f - v**RECIP_ALPHA)',
        'RECIP_TAU_O * (f * (1 - (1 - E0)**(1 / f)) * RECIP_E0'
        ' - v**RECIP_ALPHA * (q / v))'
    )
    obsrv = ('V0 * (   k1 * (1 - q     ) '
             '       + k2 * (1 - q / v ) '
             '       + k3 * (1 -     v ) )', )
    diffs = 0, 0, 0, 0
