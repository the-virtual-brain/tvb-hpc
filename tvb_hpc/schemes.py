
"""
Functions for constructing numerical schemes for nonlinear
stochastic differential equations.  Since our symbolics are
not fancy for the moment, only one-step schemes.

"""

import numpy as np
import pymbolic as pm


def euler(x, dx, dt=None):
    "Construct standard Euler step."
    dt = dt or pm.var('dt')
    return x + dt * dx


def euler_maruyama(x, f, g, dt=None, dWt=None):
    "Construct SDE Euler-Maruyama step."
    dt = dt or pm.var('dt')
    dWt = dWt or pm.var('dWt')
    return x + dt * f + dWt * dt**0.5 * g


def euler_maruyama_logp(x, f, g, xn=None, dt=None, step=euler):
    "Construct normal log p."
    dt = dt or pm.var('dt')
    mu = step(x, f, dt)
    sd = dt ** 0.5 * g
    xn = xn or np.array([pm.var(v.name + '_n') for v in x])
    return -(xn - mu) ** 2 / (2 * sd ** 2)
