import ctypes
from tvb_hpc.compiler import Compiler

code = """
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


class AlignedAlloc:
    _comp = Compiler()
    _lib = _comp(code)
    _lib.tvb_alloc_aligned.restype = ctypes.POINTER(ctypes.c_char)
    _lib.tvb_alloc_aligned.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
    _lib.tvb_free_aligned.restype = None
    _lib.tvb_free_aligned.argtypes = [ctypes.c_void_p]

    def __init__(self, size, alignment=64):
        self.size = ctypes.c_size_t(size)
        self.alignment = ctypes.c_size_t(alignment)
        self.buf = self._lib.tvb_alloc_aligned(self.size, self.alignment)
        if self.buf == 0:
            raise MemoryError("Unable to alloc %d aligned to %d" % (
                self.size.value, self.alignment.value
            ))

    def free(self):
        if hasattr(self, 'buf') and not self.buf == 0:
            self._lib.tvb_free_aligned(self.buf)
            self.buf = 0

    def __del__(self):
        self.free()
