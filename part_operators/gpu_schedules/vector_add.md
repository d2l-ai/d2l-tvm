# Vector Addition
:label:`ch_vector_add_gpu`

In this section, we will optimize the vector addition defined in :numref:`ch_vector_add` on GPU. As we seen in :numref:`ch_gpu_arch`, the overall architecture of GPUs is not significantly different to CPUs, but GPUs often have more cores and more SIMD units. If you are new to GPU programming, we recommend you to first read how we optimized the vector addition on CPUs in :numref:`ch_vector_add_cpu` for more contexts.

```{.python .input  n=22}
%matplotlib inline
import tvm
import numpy as np
import d2ltvm
import mxnet as mx
import timeit 
```

Since NumPy only supports CPU, we use MXNet  as our baseline.

```{.python .input  n=9}
sizes = 2**np.arange(10, 30, 3)

mx_add = lambda n: timeit.Timer(
    setup='import mxnet as mx\n'
    'import d2ltvm\n'
    'mx_arr = lambda x: mx.nd.array(x, ctx=mx.gpu())\n'
    'a, b, c = d2ltvm.get_abc(%d, mx_arr)' % n,
    stmt='mx.nd.elemwise_add(a, b, out=c); c.wait_to_read()')

mx_times = [d2ltvm.bench_workload(mx_add(n).timeit) for n in sizes]
mx_gflops = 2 * sizes / 1e9 / np.array(mx_times)
d2ltvm.plot_gflops(sizes, [mx_gflops], ['mxnet'])
```

## CUDA Programming 

CUDA is a programming interface proposed by Nvidia for high-performance parallel programming on GPUs. TVM provides an abstraction level above CUDA so that we don't bother to know the details about CUDA programming. We still, however, need to understand several concepts that are different to others. 

The first one is kernel. A kernel is a small program or a function. In TVM, it's often the expression to compute one element of the output tensor, e.g. `lambda i: A[i] + B[i]` in vector addition.

The second one is CUDA thread, or simply thread. A thread is abstract entity that represents the execution of a kernel. Note that it's a programming abstract, not the logic or hardware thread, which are actually running, we talked about before. So if the vector length is 1 million, then we will have 1 million thread, but apparently we cannot run all of them simultaneously. 

All these threads are grouped into blocks. If the block size is 100, then 1 million threads result in ten thousands thread blocks.  The maximal number of threads in a block is 512 before CUDA 10, and 1024 after. Each thread has a unique ID. In the 1-D indexing case, the `i`-th block is indexed by `blockIdx.x`. The number of threads in each block is `blockDim.x` and the `i`-th thread within a block is indexed by `threadIdx.x`. Therefore, the overall index of a thread can be calculated by

$$i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}.$$

It can be extended to 2-D and 3-D indexing schemes, where we add additional `.y` and `.z` fields. 

During execution, all threads in a single block will be executed in the same core, or streaming multiprocessor. We could assume them will be executed simultaneously so we can synchronize these threads in the middle of a kernel. While blocks may run on different cores. 

Analogy to CPUs, a thread block is the workload we run in parallel over CPU threads, while a thread is workload that could be vectorized. Remember in the last part of :numref:`ch_vector_add_cpu` we split the for-loop for vector addition for parallelization and vectorization separately, we can do it similarly for GPUs programming. 

```{.python .input  n=17}
factor = 64

def parallel(n):
    A, B, C = d2ltvm.vector_add(n)
    s = tvm.create_schedule(C.op)
    bx, tx = s[C].split(C.op.axis[0], factor=factor)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, (A, B, C)

s, args = parallel(256)
tvm.lower(s, args, simple_mode=True)
```

Compared to `vectorized` defined in :numref:`ch_vector_add_cpu`, there are two major difference. One is that we bind axes into block and thread indexing axes instead of call `parallel` and `vectorize`. The other one is that the pseudo codes only have the kernel, instead of the whole program with for-loops. The indexing is obtained by the 1-D indexing scheme we shown before. 

If you wrote CUDA programs before, you may want to check the generated CUDA program. Note that the target is set to be `cuda` during compiling. 

```{.python .input  n=14}
mod = tvm.build(s, args, 'cuda')
dev_mod = mod.imported_modules[0]
print(dev_mod.get_source())
```

As mentioned in :numref:`ch_gpu_arch`, GPUs have their own memories. To execute an operation on a GPU, we need to copy data from the main memory to its memory. For NVidia GPUs, the context for the $i$-th GPU can be presented by either `tvm.context('cuda', i)` or simply `tvm.gpu(i)`. The following codes copy data to the first GPU. Note the `gpu(0)` shown in the first line. 

```{.python .input}
ctx = tvm.context('cuda', 0)
tvm.nd.array(np.zeros((3,3)), ctx)
```

Remember that the `bench_vector_add_tvm` function already takes care of the context. Now let's benchmark the above schedule against MXNet. 

```{.python .input  n=16}
cuda_gflops = d2ltvm.bench_vector_add_tvm(parallel, sizes, 'cuda')
d2ltvm.plot_gflops(sizes, [mx_gflops, cuda_gflops], ['MXNet', 'TVM'])
```

We can see that TVM is faster for small workloads. That is because TVM uses `cython` for the foreign function interface, which is significantly faster than the `ctypes` used by MXNet. It's interesting to see that the simple schedule we had performs worse compared to MXNet for large vectors. It could due to the number of threads in each thread-block, which is 64, is too small to use all SIMD units in a core. Let's increase it to $256$.

```{.python .input  n=20}
factor = 256
cuda_gflops_2 = d2ltvm.bench_vector_add_tvm(parallel, sizes, 'cuda')
d2ltvm.plot_gflops(sizes, [mx_gflops, cuda_gflops, cuda_gflops_2], 
                   ['MXNet', 'TVM, block=64', 'TVM block=256'])
```

Now we can see performances are matched.

## Use OpenCL

It's recommended to use `cuda` target for Nvidia GPUs. For other GPUs, such as AMD and ARM, CUDA may be not available. But most GPUs supports [OpenCL](https://en.wikipedia.org/wiki/OpenCL).  To compile to the OpenCL backend, we only need to change the target in `tvm.build` from `cuda` to `opencl`.

```{.python .input  n=19}
opencl_gflops = d2ltvm.bench_vector_add_tvm(parallel, sizes, 'opencl')
d2ltvm.plot_gflops(sizes, [cuda_gflops_2, opencl_gflops], ['CUDA', 'OpenCL'])
```

As can been seen, the OpenCL backend has slightly larger overhead than CUDA, but their performance are identical for large workloads.

## Summary

- GPU introduces a thread-block abstraction for parallel programming. We can bind axes to threads or thread-blocks through `bind`.
- We need to assign each stream processor enough workloads for good performance.
