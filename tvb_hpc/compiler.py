
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

CFLAGS = os.environ.get('CFLAGS', '').split()
LOG.info('defaulting to %r as C flags, set CFLAGS otherwise' % CFLAGS)

LDFLAGS = os.environ.get('LDFLAGS', '').split()
LOG.info('defaulting to %r as linker flags, set LDFLAGS otherwise' % LDFLAGS)


OPENMP = True
if sys.platform == 'darwin':
    os.environ['PATH'] = '/usr/local/bin:' + os.environ['PATH']
    try:
        CC = which('gcc-6')
        CXX = which('g++-6')
        LOG.info('switched to CC=%r, CXX=%r' % (CC, CXX))
    except NoSuchExecutable:
        LOG.warning('Please brew install gcc-6 if you wish to use OpenMP.')
        OPENMP = False


class Compiler:
    """
    Handles compiler configuration & building code
    """
    source_suffix = 'c'
    default_compiler = CC

    def __init__(self, cc=None, cflags=CFLAGS, ldflags=LDFLAGS, gen_asm=False):
        self.cc = which(cc or self.default_compiler)
        self.cflags = cflags  # type: List[str]
        self.ldflags = ldflags
        self.cache = {}
        self.gen_asm = gen_asm


    def __call__(self, code: str) -> ctypes.CDLL:
        key = code, tuple(self.cflags), tuple(self.ldflags)
        if key not in self.cache:
            self.cache[key] = self._build(code)
        return self.cache[key]['dll']

    def _build(self, code):
        if not OPENMP and '-fopenmp' in self.cflags:
            self.cflags.remove('-fopenmp')
        tempdir = tempfile.TemporaryDirectory()
        c_fname = os.path.join(tempdir.name, 'code.' + self.source_suffix)
        S_fname = os.path.join(tempdir.name, 'code.S')
        obj_fname = os.path.join(tempdir.name, 'code.o')
        dll_fname = os.path.join(tempdir.name, 'code.so')
        with open(c_fname, 'w') as fd:
            fd.write(code)
        self._run([self.cc] + self.cflags + ['-fPIC', '-c', c_fname], tempdir.name)
        if self.gen_asm:
            self._run([self.cc] + self.cflags + ['-fPIC', '-S', c_fname], tempdir.name)
            with open(S_fname, 'r') as fd:
                asm = fd.read()
        self._run([self.cc] + self.cflags + self.ldflags +
                  ['-shared', obj_fname, '-o', dll_fname], tempdir.name)
        dll = ctypes.CDLL(dll_fname)
        return locals()

    def _run(self, args, cwd, **kwargs):
        LOG.info(args)
        subprocess.check_call(
            args, cwd=cwd, **kwargs
        )

class CppCompiler(Compiler):
    source_suffix = 'c++'
    default_compiler = CXX
