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


import ctypes
from .compiler import Compiler
from .utils import getLogger


posix_code = """
#include <stdlib.h>

char *tvb_alloc_aligned(size_t size, size_t alignment)
{
    void *address;
    if (!posix_memalign(&address, alignment, size))
    {
        return 0;
    }
    return ((char*) address);
}

void tvb_free_aligned(void *ptr)
{
    free(ptr);
}
"""


LOG = getLogger(__name__)


class AlignedAlloc:
    """
    Interface to ``posix_memalign`` to allocate memory with a
    given alignment.

    TODO make usuable NumPy array from allocation.

    """

    buf = None

    def __init__(self, size, alignment=64, comp: Compiler=None):
        self.size = ctypes.c_size_t(size)
        self.alignment = ctypes.c_size_t(alignment)
        self._build(comp=comp or Compiler())
        self.buf = self._lib.tvb_alloc_aligned(self.size, self.alignment)
        LOG.debug('allocd buf at %s' % (self.buf, ))
        if self.buf == 0:
            raise MemoryError("Unable to alloc %d aligned to %d" % (
                self.size.value, self.alignment.value
            ))

    def _build(self, comp: Compiler):
        _lib = comp('posix_memalign', posix_code)
        _lib.tvb_alloc_aligned.restype = ctypes.POINTER(ctypes.c_char)
        _lib.tvb_alloc_aligned.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        _lib.tvb_free_aligned.restype = None
        _lib.tvb_free_aligned.argtypes = [ctypes.c_void_p]
        self._lib = _lib
        self._comp = comp

    def free(self):
        if hasattr(self, 'buf') and not self.buf == 0:
            LOG.debug('freeing buf at %s' % (self.buf, ))
            self._lib.tvb_free_aligned(self.buf)
            self.buf = 0

    def __del__(self):
        self.free()
