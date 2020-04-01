# Depthwise Convolution
:label:`ch_depthwise_conv_gpu`

In this section, we will talk about how to optimize depthwise convolution on GPUs.

```{.python .input  n=1}
import d2ltvm
import numpy as np
import timeit
import tvm
from tvm import te

target = 'cuda'
```

## Setup

The baseline of depthwise convolution on GPUs is given by MXNet, which relies on cuDNN for high performance. Again, we benchmark the performance with various numbers of channels, when the input and kernel width/height are fixed to be 64 and 3, respectively. 
The benchmark method `depthwise_conv_timer_mxnet` has already been defined in :numref:`ch_depthwise_conv_cpu`. 
The only change that we need to make is to specify to the method that the target device is `GPU`.

```{.python .input  n=2}
channels = 2**np.arange(4, 9)
# a list of (c, n, k)
sizes = [(int(c), 64, 3) for c in channels]
mx_gflops = d2ltvm.bench_depthwise_conv_mxnet(sizes, 'gpu')
d2ltvm.plot_gflops(channels, [mx_gflops], ['mxnet'])
mx_gflops
```

It is expected to see that the performance of depthwise convoluyion on GPUs increases while the number of channels increases.

## Default schedule of Depthwise Convolution

In order to show the effectiveness of scheduling, we first apply a default schedule, which does nothing but only binds the axes to GPU thread and block.

```{.python .input  n=3}
def default_sch(ic, n, k, p, s):
    X, K, Y, PaddedX = d2ltvm.depthwise_conv(ic, n, n, k, k, p, p, s, s)
    sch = te.create_schedule(Y.op)
    sch[PaddedX].compute_inline()
    _, y, x = sch[Y].op.axis
    sch[Y].bind(y, te.thread_axis("blockIdx.x"))
    sch[Y].bind(x, te.thread_axis("threadIdx.x"))
    return sch, (X, K, Y)

default_tvm_gflops = d2ltvm.bench_depthwise_conv_tvm(default_sch, sizes, target)
d2ltvm.plot_gflops(channels, [mx_gflops, default_tvm_gflops], legend=['MXNet', 'Default_TVM'])
default_tvm_gflops
```

The default scheduling gives us the performance around 25 GFLOPS for every data shape we investigate, indicating that the compute power is not actually used.

## Scheduling of Depthwise Convolution

We work on the scheduling of depthwise convolution from the following aspects. Note that none of them is new, all covered in the previous sections.

Remember that the depthwise convolution convolves each input channel with a dedicated kernel, we can simply assign each channel to a different CUDA block. By doing this, we make different SMs work on different portions of the data, avoiding data contention across channels.

In terms of tiling, we followed the same trick done in :numref:`ch_conv_gpu` to bring some data to the shared and local memory of the GPU. You can follow the analytics approach described in :numref:`ch_conv_gpu` to calculate the cached data size and make sure the data fits in the cache. And we continue doing the cooperative fetching as before.

There is another key point for getting the good performance out of a GPU, which is mitigating the bank conflict described in :numref:`ch_conv_gpu`. Unlike using the virtual thread in :numref:`ch_conv_gpu`, this time we manipulate the data access pattern as illustrated in :numref:`fig_conv_row_column`. That is, we read the data in columns to bring them into the local memory.

```{.python .input  n=4}
tile_c = [1, 1]  # making each block take 1 channel
tile_h = [2, 8]  # making each thread take 8 rows
tile_w = [64, 1] # making each thread take 1 column

def schedule(ic, n, k, p, s):
    X, K, Y, PaddedX = d2ltvm.depthwise_conv(ic, n, n, k, k, p, p, s, s)
    sch = te.create_schedule(Y.op)
    sch[PaddedX].compute_inline()

    YL = sch.cache_write(Y, 'local')
    # create cache stage
    XX = sch.cache_read(PaddedX, 'shared', [YL])
    KK = sch.cache_read(K, 'shared', [YL])
    XL = sch.cache_read(XX, 'local', [YL])
    KL = sch.cache_read(KK, 'local', [YL])

    # tile and bind spatial axes
    c, h, w = sch[Y].op.axis
    bc, tc, ic = d2ltvm.split_axis(tile_c, sch, Y, c)
    bh, th, ih = d2ltvm.split_axis(tile_h, sch, Y, h)
    bw, tw, iw = d2ltvm.split_axis(tile_w, sch, Y, w)
    
    sch[Y].bind(bc, te.thread_axis("blockIdx.z"))
    sch[Y].bind(bh, te.thread_axis("blockIdx.y"))
    sch[Y].bind(bw, te.thread_axis("blockIdx.x"))
    sch[Y].bind(tc, te.thread_axis("threadIdx.z"))
    sch[Y].bind(th, te.thread_axis("threadIdx.y"))
    sch[Y].bind(tw, te.thread_axis("threadIdx.x"))
    sch[Y].reorder(bc, bh, bw, tc, th, tw, ic, ih, iw)

    sch[YL].compute_at(sch[Y], tw)
    
    sch[XX].compute_at(sch[Y], bw)
    sch[KK].compute_at(sch[Y], bw)
    sch[XL].compute_at(sch[Y], tw)
    sch[KL].compute_at(sch[Y], tw)
    
    # cooperative fetching
    for load in [XX, KK]:
        args = sch[load].op.axis
        fused = sch[load].fuse(*args)
        # align thread layout
        tz, fused = sch[load].split(fused, nparts=tile_c[0])
        ty, fused = sch[load].split(fused, nparts=tile_h[0])
        tx, _ = sch[load].split(fused, nparts=tile_w[0])
        sch[load].bind(tz, te.thread_axis("threadIdx.z"))
        sch[load].bind(ty, te.thread_axis("threadIdx.y"))
        sch[load].bind(tx, te.thread_axis("threadIdx.x"))
    return sch, (X, K, Y)

tvm_gflops = d2ltvm.bench_depthwise_conv_tvm(schedule, sizes, target)
d2ltvm.plot_gflops(channels, [mx_gflops, default_tvm_gflops, tvm_gflops], legend=['MXNet', 'Default_TVM', 'TVM'])
tvm_gflops
```

We can see that after properly scheduling the computation, TVM can boost the performance of depthwise convolution by over one order of magnitude, which is also much better that the MXNet baseline.

It is worthwhile noting that, in the real workloads, depthwise convolution consumes only a little computation, which can be finished in microseconds. Therefore, although TVM outperforms MXNet for quite a lot, the real executing time difference is marginal. This somewhat reflects the famous [Amdahl's law](https://en.wikipedia.org/wiki/Amdahl%27s_law), i.e. in the real use case, we should first focus on optimizing the hot spots which takes the majority of executing time.

## Summary
- Optimizing the depthwise convolution on GPUs has no major difference from optimizing other operators. The same techniques, e.g. parallelization, tiling, apply.

## Exercise
- Try to use virtual thread to mitigate the bank conflict in depthwise convolution.
- Vary the size of input data and observe the performance difference.
