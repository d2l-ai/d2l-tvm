# Vector Addition

We have already seen in :numref:`ch_vector_add_cpu` for how to optimize the CPU performance of vector addition with multi-threading. Now let's further accelerate it using GPUs. We are using `nvidia-sim` to verify we have at least one Nvidia GPU available. Note that this section executes on AMD GPUs as well.

```{.python .input  n=1}
%matplotlib inline
import tvm
import numpy as np
import d2ltvm as d2l
import mxnet as mx

!nvidia-smi
```

Again, we define a benchmark function. Compared to the version in :numref:`ch_vector_add_cpu`, here we use MXNet as the baseline. Besides, we allow to specify the device context, e.g. CPU or GPU, where the arrays will copy to.

```{.python .input  n=20}
def benchmark(func, use_tvm, ctx):
    avg_times, sizes = [], 2**np.arange(10, 30, 3)
    np.random.seed(0)
    for size in sizes:
        x = np.random.normal(size=size).astype(np.float32)
        y = np.random.normal(size=size).astype(np.float32)
        z = np.empty_like(x)
        array = tvm.nd.array if use_tvm else mx.nd.array
        x, y, z = [array(a, ctx=ctx) for a in [x, y, z]]
        res = %timeit -o -q -r3 func(x, y, z)
        avg_times.append(res.average)
    return sizes, sizes * 2 / avg_times / 1e9
```

We first use the parallel CPU scheduling introduced in :numref:`ch_vector_add_cpu` as our baseline. Let's copy the computing.

```{.python .input  n=27}
n = tvm.var('n')
A = tvm.placeholder((n,), name='a')
B = tvm.placeholder((n,), name='b')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='c')
```

Then copy the parallel scheduling and benchmark it. The target device is CPU `tvm.cpu()`, namely data will be allocated on main memory, and computation is executed on all available CPUs.

```{.python .input  n=22}
s = tvm.create_schedule(C.op)
s[C].parallel(C.op.axis[0])
mod = tvm.build(s, [A, B, C], )
_, cpu_gflops = benchmark(mod, use_tvm=True, ctx=tvm.cpu())
```

## Parallelization on GPU

GPU's thread abstraction is slightly more complex than CPU's. In GPU, threads are grouped into blocks. A thread block can execute all its threads serially or in parallel on a stream processor. The maximal number of threads in a block is 512 before CUDA 10, and 1024 after. GPU allows to use multiple thread blocks, which can be indexed by 1-D, 2-D or 3-D. Let consider the 1-D case here. In 1-D indexing, the `i`-th block is indexed by `blockIdx.x`. The number of threads in each block is `blockDim.x` and the `i`-th thread within a block is indexed by `threadIdx.x`. Therefore, the overall index of a thread can be calculated by 

$$i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}.$$

Since the vector addition only has a single for-loop, we first split into two nested for-loops.

```{.python .input  n=30}
s = tvm.create_schedule(C.op)
bx, tx = s[C].split(C.op.axis[0], factor=64)
tvm.lower(s, [A, B, C], simple_mode=True)
```

A thread-block will execute the inner loop in parallel. Since the inner loop size is 64, we are using 64 threads for each thread-block. Then different threads blocks run the outer loop in parallel. We create this parallel schedule by binding the outer loop axis `bx` to the thread-block identifier `blockIdx.x` and the inner loop axis `tx` to the thread identifier `threadIdx.x`. Then we specify the the target to `cuda` when building the module. If you programed with CUDA before, you could verify the generated CUDA codes.

```{.python .input}
s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
mod = tvm.build(s, [A, B, C], 'cuda')

dev_mod = mod.imported_modules[0]
print(dev_mod.get_source())
```

The context for the $i$-th CUDA GPU can be presented by either `tvm.context('cuda', i)` or simply `tvm.gpu(i)`. Let's benchmark the performance on the first GPU. 

```{.python .input}
_, cuda_gflops = benchmark(mod, use_tvm=True, ctx=tvm.gpu(0))
```

Comparing the CPU and GPU performance, we can see that for small workloads, e.g. length smaller than $10^5$, the performance of CPU and GPU are similar. But CPU GFLOPS is around $8$ for large vectors, while GPU GFLOPS increases to $90$. Also note this number is way below the theoretic $14899$ GFLOPS of Tesla V100 on 32-bit floating points. It is because the vector addition is memory intensive, which is hard to fully utilize the GPU computing resources. 

```{.python .input}
d2l.plot(sizes, [cpu_gflops, cuda_gflops], xlabel='Vector length', xscale='log', 
     ylabel='GFLOPS', yscale='log', legend = ['CPU', 'CUDA'])
```

Let's also compare MXNet performance on the same GPU. 

```{.python .input}
def mx_add(x, y, z):
    mx.nd.elemwise_add(x, y, out=z)
    z.wait_to_read()
    
_, mx_gflops = benchmark(mx_add, use_tvm=False, ctx=mx.gpu(0))

d2l.plot(sizes, [cuda_gflops, mx_gflops], xlabel='Vector length', xscale='log', 
         ylabel='GFLOPS', yscale='log', legend = ['TVM CUDA', 'MXNet'])
```

We can see that TVM is faster for small workloads. That is because TVM uses `cython` for the foreign function interface, which is significantly faster than the `ctypes` used by MXNet. It's interesting to see that the simple schedule we had performs worse compared to MXNet for large vectors. It could due to the number of threads in each thread-block, which is 64, is too small to use all cores in a stream processor. Let's increase it to $256$. 

```{.python .input}
s = tvm.create_schedule(C.op)
bx, tx = s[C].split(C.op.axis[0], factor=256)
s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
mod = tvm.build(s, [A, B, C], 'cuda')
_, cuda_gflops_2 = benchmark(mod, use_tvm=True, ctx=tvm.gpu(0))
```

```{.python .input}
d2l.plot(sizes, [cuda_gflops, mx_gflops, cuda_gflops_2], 
         xlabel='Vector length', xscale='log', ylabel='GFLOPS', yscale='log', 
         legend = ['CUDA, 64', 'MXNet', 'CUDA 256'], fmts=['--', '--', '-'])
```

Now we can see performances are matched. 

## Use OpenCL

It's recommended to use `cuda` target for Nvidia GPUs. For other GPUs, such as AMD and ARM, CUDA is maybe not available. In most cases, however, OpenCL is supported. We only need to change the target in `tvm.build` to `opencl` and create an `opencl` device for it.

```{.python .input  n=5}
s = tvm.create_schedule(C.op)
bx, tx = s[C].split(C.op.axis[0], factor=256)
s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
mod = tvm.build(s, [A, B, C], 'opencl')

_, opencl_gflops = benchmark(mod, use_tvm=True, ctx=tvm.context('opencl', 0))

d2l.plot(sizes, [cuda_gflops_2, opencl_gflops], xlabel='Vector length', 
         xscale='log',  ylabel='GFLOPS', yscale='log', 
         legend=['CUDA', 'OpenCL'])
```

As can been seen, the OpenCL backend has slightly larger overhead than CUDA, but their performance are identical for large workloads.

## Summary

- GPU introduces a thread-block abstraction for parallel programming. We can bind axes to threads or thread-blocks through `bind`.
- We need to assign each stream processor enough workloads for good performance.
