
import os
import os.path
import logging
import ctypes
import tempfile
import subprocess


LOG = logging.getLogger(__name__)


CC = os.environ.get('CC', 'gcc')
LOG.info('defaulting to %r as C compiler, set CC otherwise' % CC)


CFLAGS = os.environ.get('CFLAGS', '').split()
LOG.info('defaulting to %r as C flags, set CFLAGS otherwise' % CFLAGS)


LDFLAGS = os.environ.get('LDFLAGS', '').split()
LOG.info('defaulting to %r as linker flags, set LDFLAGS otherwise' % LDFLAGS)


class NoSuchExecutable(RuntimeError):
    pass


def which(exe):
    if os.path.exists(exe):
        return exe
    for path in os.environ['PATH'].split(os.path.pathsep):
        maybe_path = os.path.join(path, exe)
        if os.path.exists(maybe_path):
            return maybe_path
    raise NoSuchExecutable(exe)


class Compiler:
    """
    Handles compiler configuration & building code
    """

    def __init__(self, cc=CC, cflags=CFLAGS, ldflags=LDFLAGS):
        self.cc = which(cc)
        self.cflags = cflags
        self.ldflags = ldflags
        self.cache = {}

    def __call__(self, code: str) -> ctypes.CDLL:
        key = code, tuple(self.cflags), tuple(self.ldflags)
        if key not in self.cache:
            self.cache[key] = self._build(code)
        return self.cache[key]['dll']

    def _build(self, code):
        tempdir = tempfile.TemporaryDirectory()
        c_fname = os.path.join(tempdir.name, 'code.c')
        obj_fname = os.path.join(tempdir.name, 'code.o')
        dll_fname = os.path.join(tempdir.name, 'code.so')
        with open(c_fname, 'w') as fd:
            fd.write(code)
        run = subprocess.check_call
        self._run([CC] + CFLAGS + ['-fPIC', '-c', c_fname], tempdir.name)
        self._run([CC] + CFLAGS + LDFLAGS +
                  ['-shared', obj_fname, '-o', dll_fname], tempdir.name)
        dll = ctypes.CDLL(dll_fname)
        return locals()

    def _run(self, args, cwd, **kwargs):
        subprocess.check_call(
            args, cwd=cwd, **kwargs
        )
