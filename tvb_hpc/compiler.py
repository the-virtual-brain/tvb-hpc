
"""
The compiler module handles invocation of compilers to generate a shared lib
which can be loaded via ctypes.

"""

import os
import os.path
import sys
import logging
import ctypes
import tempfile
import subprocess
from tvb_hpc.utils import which, NoSuchExecutable


LOG = logging.getLogger(__name__)


CC = os.environ.get('CC', 'gcc')
LOG.info('defaulting to %r as C compiler, set CC otherwise' % CC)

CXX = os.environ.get('CXX', 'g++')
LOG.info('defaulting to %r as C compiler, set CXX otherwise' % CXX)

CFLAGS = os.environ.get('CFLAGS', '-std=c99 -Wall -Wextra').split()
LOG.info('defaulting to %r as C flags, set CFLAGS otherwise' % CFLAGS)

CXXFLAGS = os.environ.get('CXXFLAGS', '-std=c++11 -Wall -Wextra').split()
LOG.info('defaulting to %r as CXX flags, set CXXFLAGS otherwise' % CXXFLAGS)

LDFLAGS = os.environ.get('LDFLAGS', '').split()
LOG.info('defaulting to %r as linker flags, set LDFLAGS otherwise' % LDFLAGS)


OPENMP = False
if sys.platform == 'darwin':
    os.environ['PATH'] = '/usr/local/bin:' + os.environ['PATH']
    try:
        CC = which('gcc-6')
        CXX = which('g++-6')
        LOG.info('switched to CC=%r, CXX=%r' % (CC, CXX))
    except NoSuchExecutable:
        LOG.warning('Please brew install gcc-6 if you wish to use OpenMP.')


class Compiler:
    """
    Handles compiler configuration & building code

    """
    source_suffix = 'c'
    default_compiler = CC
    default_compiler_flags = CFLAGS

    def __init__(self, cc=None, cflags=None, ldflags=LDFLAGS, gen_asm=False,
                 openmp=OPENMP):
        """

        :param cc: compiler to use
        :param cflags: list of flags to pass to compiler
        :param ldflags: list of linker flags
        :param gen_asm: bool, generate and store assembly
        :param openmp: bool, use OpenMP

        """
        self.cc = which(cc or self.default_compiler)
        self.cflags = cflags or self.default_compiler_flags
        self.ldflags = ldflags
        self.cache = {}
        self.gen_asm = gen_asm
        self.openmp = openmp
        if self.openmp:
            self.cflags.append('-fopenmp')

    def __call__(self, name: str, code: str) -> ctypes.CDLL:
        key = name
        if key not in self.cache:
            self.cache[key] = self._build(name, code)
        return self.cache[key]['dll']

    def _build(self, name, code):
        LOG.debug('compiling unit %r with code\n%s' % (name, code))
        tempdir = tempfile.TemporaryDirectory()
        c_fname = os.path.join(tempdir.name, name + '.' + self.source_suffix)
        S_fname = os.path.join(tempdir.name, name + '.S')
        obj_fname = os.path.join(tempdir.name, name + '.o')
        dll_fname = os.path.join(tempdir.name, name + '.so')
        with open(c_fname, 'w') as fd:
            fd.write(code)
        self._run([self.cc] + self.cflags + ['-fPIC', '-c', c_fname],
                  tempdir.name)
        if self.gen_asm:
            self._run([self.cc] + self.cflags
                      + ['-fverbose-asm', '-S', c_fname], tempdir.name)
            with open(S_fname, 'r') as fd:
                asm = fd.read()
        self._run([self.cc] + self.cflags + self.ldflags +
                  ['-shared', obj_fname, '-o', dll_fname], tempdir.name)
        dll = ctypes.CDLL(dll_fname)
        return locals()

    def _run(self, args, cwd, **kwargs):
        LOG.debug(args)
        subprocess.check_call(
            args, cwd=cwd, **kwargs
        )


class CppCompiler(Compiler):
    source_suffix = 'c++'
    default_compiler = CXX
    default_compiler_flags = CXXFLAGS
