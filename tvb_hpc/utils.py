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

import enum
import os.path
import numpy as np
import logging
import time
import pymbolic as pm
from pymbolic import parse
from pymbolic.mapper.stringifier import SimplifyingSortingStringifyMapper
from pymbolic.mapper import IdentityMapper
from sympy.parsing.sympy_parser import parse_expr


here = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.normpath(os.path.join(here, '..', 'include'))


def can_bcast(n, m):
    return (n == 1) or (m == 1) or (n == m)


def getLogger(name):
    return logging.getLogger(name)


class NoSuchExecutable(RuntimeError):
    "Exception raised when ``which`` does not find its argument on $PATH."
    pass


def which(exe):
    "Find exe name on $PATH."
    if os.path.exists(exe):
        return exe
    for path in os.environ['PATH'].split(os.path.pathsep):
        maybe_path = os.path.join(path, exe)
        if os.path.exists(maybe_path):
            return maybe_path
    raise NoSuchExecutable(exe)


def simplify(expr):
    "Simplify pymbolic expr via SymPy."
    # TODO switch entirely to SymPy?
    parts = SimplifyingSortingStringifyMapper()(expr)
    simplified = parse_expr(parts).simplify()
    return parse(repr(simplified))


def vars(svars):
    """
    Build array of symbolic variables from string of space-separated variable
    names.

    """
    return np.array([pm.var(var) for var in svars.split()])


def exprs(sexprs):
    """
    Build array of symbolic expresssions from sequence of strings.

    """
    exprs = []
    for expr in sexprs:
        if isinstance(expr, (int, float)):
            exprs.append(expr)
        else:
            try:
                exprs.append(pm.parse(expr))
            except Exception as exc:
                raise Exception(repr(expr))
    return np.array(exprs)


class timer:
    logger = getLogger('tvb_hpc.utils.timer')

    def __init__(self, msg=''):
        self.msg = msg

    def __enter__(self, *args):
        self.tic = time.time()
        return self

    def __exit__(self, *args):
        toc = time.time()
        self.elapsed = toc - self.tic
        msg = '%s elapsed %.3fs'
        self.logger.info(msg, self.msg, self.elapsed)



class VarSubst(IdentityMapper):
    "Substitute variables by name for expressions."

    def __init__(self, verbose=False, **substs):
        self.verbose = verbose
        self.substs = substs

    def map_variable(self, var: pm.var, *args, **kwargs):
        if self.verbose:
            print(var.name)
        return self.substs.get(var.name, var)


def subst_vars(expr, **var_name_to_exprs):
    expr = pm.parse(str(expr))
    return VarSubst(**var_name_to_exprs)(expr)


def scaling(ary):
    "Heuristic for linear vs log scaling of array."
    lin, _ = np.histogram(ary.flat[:], 5)
    log, _ = np.histogram(np.log(ary.flat[:] + 1), 5)
    if np.std(lin/lin.max()) > np.std(log/log.max()):
        return 'log'
    return 'linear'


def loadtxt_many(fnames):
    "Multiprocess np.loadtxt.  If fnames is str, glob it."
    if isinstance(fnames, str):
        import glob
        fnames = glob.glob(fnames)
    import multiprocessing, numpy
    pool = multiprocessing.Pool()
    arrays = pool.map(numpy.loadtxt, fnames)
    pool.close()
    return arrays