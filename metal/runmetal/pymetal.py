import io
import logging
import os
import subprocess
from typing import List

import Foundation
import numpy
import objc

from . import Metal

log = logging.getLogger(__name__)


class PyMetal:
    PixelFormatRGBA8UNorm = Metal.MTLPixelFormatRGBA8Unorm
    StorageModeManaged = Metal.MTLResourceStorageModeManaged

    def __init__(self):
        self.dev = None

    def setopt(self, vv, opts: dict):
        for k, v in opts.items():
            fn = "set" + k + "_"
            if hasattr(vv, fn):
                log.debug("set(%s) %s <- %s", type(vv).__name__, k, v)
                getattr(vv, fn)(v)
            else:
                log.error("cannot set(%s) <- %s (%s not found)",
                          type(vv).__name__, k, v, fn)
        return vv

    def maxvalues(self) -> dict:
        res = {}
        for k in filter(lambda f: f.startswith("max") and not f.endswith("_"), dir(self.dev)):
            name = k[3:]
            res[name] = getattr(self.dev, k)()
        return res

    def configs(self) -> dict:
        res = {}
        for k in filter(lambda f: f.endswith("Config"), dir(self.dev)):
            name = k[:-6]
            res[name] = getattr(self.dev, k)()
        return res

    def logmethods(self, obj, pat=None):
        if pat is None:
            res = dir(obj)
        else:
            res = list(filter(lambda f: f.find(pat) != -1, dir(obj)))
        log.debug("%s %s %s", type(obj), pat, res)
        return res

    def lsdev(self):
        return Metal.MTLCopyAllDevices()

    def device2str(self, d) -> List[str]:
        def yes(x):
            if x:
                return "yes"
            return "no"

        def supported(x):
            if x:
                return "✅ supported"
            return "❌ not supported"

        res = []
        res.append(d.name() + ":")
        res.append("	• low-power: " + yes(d.isLowPower()))
        res.append("	• removable: " + yes(d.isRemovable()))
        res.append("	• configured as headless: " + yes(d.isHeadless()))
        res.append("	• registry ID: " + str(d.registryID()))
        res.append("")
        res.append("	Feature Sets:")
        for k in filter(lambda f: f.startswith("MTLFeatureSet_"), dir(Metal)):
            name = k[14:]
            val = getattr(Metal, k)
            res.append("	• %s: %s" %
                       (name, supported(d.supportsFeatureSet_(val))))
        return res

    def opendevice(self, name=None):
        if name is None:
            self.dev = Metal.MTLCreateSystemDefaultDevice()
        else:
            devs = list(filter(lambda f: f.name() == name, self.lsdev()))
            if len(devs) == 1:
                self.dev = devs[0]
            else:
                raise Exception("no such device: %s / %s" %
                                (name, list(map(lambda f: f.name(), self.lsdev()))))

    def openlibrary(self, src=None, filename=None, **kwargs):
        if filename is not None:
            if isinstance(filename, str):
                src = open(filename).read()
            elif isinstance(filename, io.IOBase):
                src = filename.read()
            elif isinstance(filename, (list, tuple)):
                src = "\n".join(map(lambda f: open(f).read(), filename))
        opts = Metal.MTLCompileOptions.new()
        self.setopt(opts, kwargs)
        # err = Foundation.NSError.alloc()
        log.debug("openlibrary(source)")
        self.lib = self.dev.newLibraryWithSource_options_error_(
            src, opts, objc.NULL)[0]
        if self.lib is None:
            log.error("compile error?: %s", src)

    def openlibrary_compiled(self, filename, **kwargs):
        log.debug("openlibrary(compiled): %s", filename)
        self.lib = self.dev.newLibraryWithFile_error_(filename, objc.NULL)
        if self.lib is None:
            log.error("load error?: %s", filename)

    def openlibrary_default(self, **kwargs):
        log.debug("openlibrary(default)")
        self.lib = self.dev.newDefaultLibrary()
        if self.lib is None:
            log.error("load error?(default)")

    def getfn(self, name):
        return self.lib.newFunctionWithName_(name)

    def emptybuffer(self, size, label=None, opts=0):
        res = self.dev.newBufferWithLength_options_(size, opts)
        if label is not None:
            res.setLabel_(label)
        return res

    def numpybuffer(self, data, label=None, opts=0):
        buf = self.emptybuffer(data.nbytes, label, opts)
        buf.contents().as_buffer(buf.length())[:] = data.tobytes()
        return buf

    def bytesbuffer(self, data, label=None, opts=0):
        res = self.dev.newBufferWithLength_options_(len(data), opts)
        if label is not None:
            res.setLabel_(label)
        ibc = res.contents().as_buffer(res.length())
        ibc[:] = data
        return res

    def intbuffer(self, i):
        return self.numpybuffer(numpy.array(i, dtype=numpy.int32))

    def uintbuffer(self, i):
        return self.numpybuffer(numpy.array(i, dtype=numpy.uint32))

    def floatbuffer(self, f):
        return self.numpybuffer(numpy.array(f, dtype=numpy.float32))

    def syncbuffer(self, buffer, start=0, length=None):
        # sync from CPU to GPU
        if length is None:
            length = buffer.length() - start
        buffer.didModifyRange_(Foundation.NSRange(
            location=start, length=length))

    def emptytexture(self, size, label=None, opts=0):
        desc = Metal.MTLTextureDescriptor.new()
        log.info("texture desc: %s", desc)
        # TODO

    def buf2byte(self, buf):
        return buf.contents().as_buffer(buf.length())

    def buf2numpy(self, buf, dtype):
        return numpy.frombuffer(buf.contents().as_buffer(buf.length()), dtype=dtype)

    def getqueue(self, **kwargs):
        cqueue = self.dev.newCommandQueue()
        self.setopt(cqueue, kwargs)
        cbuffer = cqueue.commandBuffer()
        self.setopt(cbuffer, kwargs)
        return cqueue, cbuffer

    def getmtlsize(self, arg):
        if isinstance(arg, int):
            return Metal.MTLSize(width=arg, height=1, depth=1)
        return Metal.MTLSize(**arg)

    def enqueue_compute(self, cbuffer, func, buffers, threads=None, iters=None, label=None):
        desc = Metal.MTLComputePipelineDescriptor.new()
        if label is not None:
            desc.setLabel_(label)
        desc.setComputeFunction_(func)
        state = self.dev.newComputePipelineStateWithDescriptor_error_(
            desc, objc.NULL)
        encoder = cbuffer.computeCommandEncoder()
        encoder.setComputePipelineState_(state)
        bufmax = 0
        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)
            if bufmax < buf.length():
                bufmax = buf.length()
        if iters is not None:
            bufmax = iters
        # threads
        if threads is None:
            # number of thread per group
            w = state.threadExecutionWidth()
            h = max(1, int(state.maxTotalThreadsPerThreadgroup() / w))
            log.debug("w,h=%d,%d, bufmax=%d", w, h, bufmax)
            tpg = self.getmtlsize({"width": w, "height": h, "depth": 1})
            # number of thread group per grid
            # w2 = max(1, int((bufmax + w * h - 1) / (w * h)))
            w2 = int(max(1, (bufmax + w - 1) / w))
            ntg = self.getmtlsize(w2)
            log.debug("threads: ntg=%s, tpg=%s", ntg, tpg)
            # encoder.dispatchThreadgroups_threadsPerThreadgroup_(ntg, tpg)
            encoder.dispatchThreads_threadsPerThreadgroup_(ntg, tpg)
        else:
            assert len(threads) >= 2
            # number of thread group
            ntg = Metal.MTLSize(width=threads[0], height=threads[1], depth=1)
            # number of thread per group
            _w = state.threadExecutionWidth()
            _h = state.maxTotalThreadsPerThreadgroup() / _w
            tpg = Metal.MTLSize(width=_w, height=_h, depth=1)
            log.debug("threads: %s %s", ntg, tpg)
            encoder.dispatchThreads_threadsPerThreadgroup_(ntg, tpg)
        log.debug("encode(compute) %s", label)
        encoder.endEncoding()

    def runThread2d(self, w, h):
        '''
        MTLSize threadsPerGrid = MTLSizeMake(width, height, 1);
  NSUInteger _w = pipeline.threadExecutionWidth;
  NSUInteger _h = pipeline.maxTotalThreadsPerThreadgroup / _w;
  MTLSize threadsPerThreadgroup = MTLSizeMake(_w, _h, 1);
  [commandEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        '''
        # self.dev.runThreadsWidth(w, h)
        threadsPerGrid = self.getmtlsize(w, h, 1)
        pipeline = Metal.MTLComputePipelineDescriptor.new()
        _w = pipeline.threadExecutionWidth
        _h = pipeline.maxTotalThreadsPerThreadgroup / _w
        threadsPerThreadgroup = self.getmtlsize(_w, _h, 1)

    def runThread(self, cbuffer, func, buffers, threads=None, label=None):
        desc = Metal.MTLComputePipelineDescriptor.new()
        if label is not None:
            desc.setLabel_(label)
        desc.setComputeFunction_(func)
        state = self.dev.newComputePipelineStateWithDescriptor_error_(
            desc, objc.NULL)
        encoder = cbuffer.computeCommandEncoder()
        encoder.setComputePipelineState_(state)
        bufmax = 0
        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)
            if bufmax < buf.length():
                bufmax = buf.length()

        # threads

        # number of thread per group
        w = state.threadExecutionWidth()
        h = max(1, int(state.maxTotalThreadsPerThreadgroup() / w))
        log.debug("w,h=%d,%d, bufmax=%d", w, h, bufmax)
        tpg = self.getmtlsize({"width": w, "height": h, "depth": 1})

        # number of thread per grig
        ntg = self.getmtlsize(threads)
        log.debug("threads: %s %s", ntg, tpg)

        encoder.dispatchThreads_threadsPerThreadgroup_(ntg, tpg)
        log.debug("encode(compute) %s", label)
        encoder.endEncoding()

    def enqueue_blit(self, cbuffer, texture=None, label=None):
        encoder = cbuffer.blitCommandEncoder()
        if label is not None:
            encoder.setLabel_(label)
        if texture is not None:
            encoder.synchronizeResource_(texture)
        log.debug("encode(blit) %s", label)
        encoder.endEncoding()

    def enqueue_render(self, cbuffer, buffers):
        # TBD
        log.error("enqueue_render: not implemented")
        pass

    def start_process(self, cbuffer):
        log.debug("start compute")
        cbuffer.commit()

    def wait_process(self, cbuffer):
        log.debug("wait")
        cbuffer.waitUntilCompleted()
        log.debug("finished")

    def compile(self, source, outfn):
        basepath = "/Applications/Xcode.app/Contents/Developer/Toolchains"
        metalpath = os.path.join(basepath, "XcodeDefault.xctoolchain",
                                 "usr", "metal", "macos", "bin", "metal")
        cmd = [metalpath, "-x", "metal", "-", "-o", outfn]
        log.debug("compile: %s", cmd)
        p1 = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        p1.communicate(source.encode("utf-8"))
