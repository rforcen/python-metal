import random
import re

import numba
import numpy

import runmetal


class Metal():
    def __init__(self, source, func_name=None):
        self.pm = runmetal.PyMetal()
        self.pm.opendevice()
        if source.find('.metallib') != -1:
            self.pm.openlibrary_compiled(source)
        elif source.find('.metal') != -1:
            self.pm.openlibrary(src=None, filename=source)
        elif source.find('kernel'):
            self.pm.openlibrary(source)

        if func_name is not None:
            self.fn = self.pm.getfn(func_name)
        else:  # search first func kernel void func_name
            self.fn = self.pm.getfn(re.search("kernel\s+void\s+(\w+)", source)[1])

        self.cqueue, self.cbuffer, self.buffer_list = None, None, None

    @staticmethod
    @numba.njit(cache=True, parallel=True)
    def rand(n, dtype):
        v = numpy.empty(n, dtype=dtype)
        for i in numba.prange(n):
            v[i] = random.uniform(0, 1)
        return v

    # replace search_str for rpl_str in input file file_in generating file_out
    @staticmethod
    def file_replace(file_in, file_out, search_str, rpl_str):
        with open(file_in, 'r') as file: file_data = file.read()
        kernel_func = re.search("kernel\s+void\s+(\w+)", file_data)[1]
        with open(file_out, 'w') as file: file.write(file_data.replace(search_str, rpl_str))
        return file_out, kernel_func

    def buffer(self, data):
        if type(data).__module__ == numpy.__name__:
            return self.pm.numpybuffer(data)
        if type(data) is int:
            return self.pm.intbuffer(data)
        if type(data) is float:
            return self.pm.floatbuffer(data)

    def float_buf(self, data):
        return self.buffer(numpy.array(data, dtype=numpy.float32))

    def int_buf(self, data):
        return self.buffer(numpy.array(data, dtype=numpy.int32))

    def empty(self, size):
        return self.pm.emptybuffer(size)

    def empty_int(self, size):
        return self.empty(size * numpy.dtype(numpy.int32).itemsize)

    def empty_float(self, size):
        return self.empty(size * numpy.dtype(numpy.float32).itemsize)

    def set_buffers(self, buffers=None, threads=None):  # set of buffers
        self.cqueue, self.cbuffer = self.pm.getqueue()
        self.buffer_list = buffers
        self.pm.enqueue_compute(cbuffer=self.cbuffer, func=self.fn, threads=threads, buffers=buffers)

    def run(self):
        self.pm.start_process(self.cbuffer)
        self.pm.wait_process(self.cbuffer)

    def get_buffer(self, buf, dtype):
        return self.pm.buf2numpy(buf, dtype)
