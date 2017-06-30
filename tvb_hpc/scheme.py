#     Copyright 2017 TVB-HPC contributors
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
Functions for constructing numerical schemes for nonlinear
stochastic differential equations.  Since our symbolics are
not fancy for the moment, only one-step schemes.

"""

from typing import Union
import numpy as np
import pymbolic as pm
from .base import BaseKernel


def euler(x, f, dt=None):
    "Construct standard Euler step."
    dt = dt or pm.var('dt')
    return x + dt * f


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


class TimeStepScheme(BaseKernel):
    pass


class EulerStep(TimeStepScheme):

    # TODO merge with model dfun kernel to do multiple steps

    def __init__(self, dt: Union[pm.var, float]):
        self.dt = dt

    def kernel_dtypes(self):
        dtypes = {
            'nnode,nsvar': np.uintc,
            'state,drift': np.float32,
        }
        if isinstance(self.dt, pm.var):
            dtypes['dt'] = np.float32
        return dtypes

    def kernel_data(self):
        data = 'nnode nsvar state drift'.split()
        if isinstance(self.dt, pm.var):
            data.insert(0, self.dt.name)
        return data

    def kernel_domains(self):
        domains = "[i_node, i_svar]"
        constraints = "0 <= i_node < nnode and 0 <= i_svar < nsvar"
        return f"{{ {domains}: {constraints} }}"

    def kernel_isns(self):
        lhs = 'state[i_node, i_svar]'
        rhs = f'state[i_node, i_svar] + {self.dt} * drift[i_node, i_svar]'
        return [f'{lhs} = {rhs}']


class EulerMaryuyamaStep(EulerStep):

    def kernel_isns(self):
        lhs = 'state[i_node, i_svar] = '
        rhs = 'state[i_node, i_svar] + {dt}*drift[i_node, i_svar] '\
              '+ sqrt_dt*dWt[i_node, i_svar]*diffs[i_node, i_svar]'
        return [
            '<> sqrt_dt = sqrtf({dt})'.format(dt=self.dt),
            (lhs + rhs).format(dt=self.dt)
        ]

    def kernel_data(self):
        data = super().kernel_data()
        data += ['dWt', 'diffs']
        return data

    def kernel_dtypes(self):
        dtypes = super().kernel_dtypes()
        dtypes['dWt,diffs'] = np.float32
        return dtypes
