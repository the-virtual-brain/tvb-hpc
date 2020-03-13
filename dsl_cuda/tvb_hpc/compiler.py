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
The compiler module handles invocation of compilers to generate a shared lib
which can be loaded via ctypes.

"""

import os
import os.path
import logging
import ctypes
import tempfile
import weakref
import subprocess
from typing import List
import numpy as np
import loopy as lp
from loopy.target.c import CTarget, generate_header, CASTBuilder
import cgen
from dsl_cuda.tvb_hpc import which


LOG = logging.getLogger(__name__)


class Spec:
    """
    Spec handles details dtype, vectorization, alignment, etc which affect
    code generation but not the math.

    """

    def __init__(self, float='float', width=8, align=None, openmp=False,
                 layout=None, debug=False):
        self.float = float
        self.width = width
        self.align = align
        self.openmp = openmp
        # TODO refactor this
        self.layout = layout
        self.debug = debug

    # TODO refactor using loopy et al
    @property
    def dtype(self):
        return self.float

    @property
    def np_dtype(self):
        return {'float': np.float32}[self.dtype]

    @property
    def ct_dtype(self):
        import ctypes as ct
        return {'float': ct.c_float}[self.dtype]

    # TODO refactor so not needed
    @property
    def dict(self):
        return {
            'float': self.float,
            'width': self.width,
            'align': self.align,
            'openmp': self.openmp
        }


class Compiler:
    """
    Wraps a C compiler to build and load shared libraries.

    """

    source_suffix = 'c'
    default_exe = 'gcc'
    default_compile_flags = '-std=c99 -g -O3 -fPIC'.split()
    default_link_flags = '-shared'.split()

    def __init__(self, cc: str=None,
                 cflags: List[str]=None,
                 ldflags: List[str]=None):
        self.exe = which(cc) if cc else self.default_exe
        self.cflags = cflags or self.default_compile_flags[:]
        self.ldflags = ldflags or self.default_link_flags[:]
        self.tempdir = tempfile.TemporaryDirectory()

    def _tempname(self, name):
        "Build temporary filename path in tempdir."
        return os.path.join(self.tempdir.name, name)

    def _call(self, args, **kwargs):
        "Invoke compiler with arguments."
        cwd = self.tempdir.name
        args_ = [self.exe] + args
        LOG.debug(args_)
        subprocess.check_call(args_, cwd=cwd, **kwargs)

    def build(self, code: str) -> ctypes.CDLL:
        "Compile code, build and load shared library."
        LOG.debug(code)
        c_fname = self._tempname('code.' + self.source_suffix)
        obj_fname = self._tempname('code.o')
        dll_fname = self._tempname('code.so')
        with open(c_fname, 'w') as fd:
            fd.write(code)
        self._call(self.compile_args(c_fname))
        self._call(self.link_args(obj_fname, dll_fname))
        return ctypes.CDLL(dll_fname)

    def compile_args(self, c_fname):
        "Construct args for compile command."
        return self.cflags + ['-c', c_fname]

    def link_args(self, obj_fname, dll_fname):
        "Construct args for link command."
        return self.ldflags + ['-shared', obj_fname, '-o', dll_fname]


class CppCompiler(Compiler):
    "Subclass of Compiler to invoke a C++ compiler."
    source_suffix = 'c++'
    default_exe = 'g++'
    default_compile_flags = '-g -O3'.split()


class CompiledKernel:
    """
    A CompiledKernel wraps a loop kernel, compiling it and loading the
    result as a shared library, and provides access to the kernel as a
    ctypes function object, wrapped by the __call__ method, which attempts
    to automatically map argument types.

    """

    def __init__(self, knl: lp.LoopKernel, comp: Compiler=None):
        assert isinstance(knl.target, CTarget)
        self.knl = knl
        self.code, _ = lp.generate_code(knl)
        self.comp = comp or Compiler()
        self.dll = self.comp.build(self.code)
        self.func_decl, = generate_header(knl)
        self._arg_info = []
        # TODO knl.args[:].dtype is sufficient
        self._visit_func_decl(self.func_decl)
        self.name = self.knl.name
        restype = self.func_decl.subdecl.typename
        if restype == 'void':
            self.restype = None
        else:
            raise ValueError('Unhandled restype %r' % (restype, ))
        self._fn = getattr(self.dll, self.name)
        self._fn.restype = self.restype
        self._fn.argtypes = [ctype for name, ctype in self._arg_info]
        self._prepared_call_cache = weakref.WeakKeyDictionary()

    def __call__(self, **kwargs):
        "Execute kernel with given args mapped to ctypes equivalents."
        args_ = []
        for knl_arg, arg_t in zip(self.knl.args, self._fn.argtypes):
            arg = kwargs[knl_arg.name]
            if hasattr(arg, 'ctypes'):
                if arg.size == 0:
                    # TODO eliminate unused arguments from kernel
                    arg_ = arg_t(0.0)
                else:
                    arg_ = arg.ctypes.data_as(arg_t)
            else:
                arg_ = arg_t(arg)
            args_.append(arg_)
        self._fn(*args_)

    def _append_arg(self, name, dtype, pointer=False):
        "Append arg info to current argument list."
        self._arg_info.append((
            name,
            self._dtype_to_ctype(dtype, pointer=pointer)
        ))

    def _visit_const(self, node: cgen.Const):
        "Visit const arg of kernel."
        if isinstance(node.subdecl, cgen.RestrictPointer):
            self._visit_pointer(node.subdecl)
        else:
            pod = node.subdecl  # type: cgen.POD
            self._append_arg(pod.name, pod.dtype)

    def _visit_pointer(self, node: cgen.RestrictPointer):
        "Visit pointer argument of kernel."
        pod = node.subdecl  # type: cgen.POD
        self._append_arg(pod.name, pod.dtype, pointer=True)

    def _visit_func_decl(self, func_decl: cgen.FunctionDeclaration):
        "Visit nodes of function declaration of kernel."
        for i, arg in enumerate(func_decl.arg_decls):
            if isinstance(arg, cgen.Const):
                self._visit_const(arg)
            elif isinstance(arg, cgen.RestrictPointer):
                self._visit_pointer(arg)
            else:
                raise ValueError('unhandled type for arg %r' % (arg, ))

    def _dtype_to_ctype(self, dtype, pointer=False):
        "Map NumPy dtype to equivalent ctypes type."
        target = self.knl.target  # type: CTarget
        registry = target.get_dtype_registry().wrapped_registry
        typename = registry.dtype_to_ctype(dtype)
        typename = {'unsigned': 'uint'}.get(typename, typename)
        basetype = getattr(ctypes, 'c_' + typename)
        if pointer:
            return ctypes.POINTER(basetype)
        return basetype


class OpenMPCASTBuilder(CASTBuilder):

    def emit_sequential_loop(self, codegen_state, iname, iname_dtype,
                             lbound, ubound, inner):
        from cgen import Pragma, Block
        loop = super().emit_sequential_loop(codegen_state, iname, iname_dtype,
                                            lbound, ubound, inner)

        pragma = self.target.iname_pragma_map.get(iname)
        if pragma:
            return Block(contents=[
                Pragma(pragma),
                loop,
            ])
        return loop


class OpenMPCTarget(CTarget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iname_pragma_map = {}

    def get_device_ast_builder(self):
        return OpenMPCASTBuilder(self)
