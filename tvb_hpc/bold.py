
"""
Source for Balloon-Windkessel model of BOLD.

"""

from tvb_hpc.model import BaseModel


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
    return {k: v for k, v in locals().items() if isinstance(v, float)}


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

