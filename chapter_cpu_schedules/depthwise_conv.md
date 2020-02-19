# Depthwise Convolution
:label:`ch_depthwise_conv_cpu`

In this section, we will follow the packing idea presented in :numref:`ch_packed_conv_cpu` to re-defined the computation of depthwise convolution and schedule it to run efficiently on CPUs. Similar to the 2-D convolution, tiling the data along the channel dimension and packing it into `NCHW[x]c` benefit the performance significantly.

```{.python .input  n=3}
import d2ltvm
import numpy as np
import tvm

#target = 'llvm -mcpu=skylake-avx512'
target = 'llvm -mcpu=core-avx2'
```

## Packing Data and Weight

Recall the depthwise convolution described in :numref"`ch_depthwise_conv`, it differs from the 2-D convolution by having each channel of the input data convolved with a separated kernel. Therefore, the packing mechanism of input data is exactly the same as we did in :numref:`ch_packed_conv_cpu`. Kernel is a bit different, as the size is in `[oc, 1, kh, kw]`, which means that there is no need to tile the input channel.

we first tiled the width and height axes and then moved the inner axes as the innermost dimensions during computing. It's the same idea to tile the channels. But we do one additional step to actually move the data in the inner channel loop to the last dimension, i.e. do data layout transformation. So we pay the data movement cost once, while improving the data accessing performance significantly.

The following code block splits the channel dimension and move the data in NumPy.

Now let's implement the pack computation in TVM.

```{.python .input  n=4}
def depthwise_conv_pack(c, nh, nw, kh, kw, ph, pw, tc):
    """Pack data and weight for depthwise convolution
       Note that the input channel of kernel is specified as 1,
       and the output channel of kernel equals the input channel of data

    c : input channel of data and output channel of kernel
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding
    tc : the tiling size of channels
    """
    X = tvm.placeholder((c, nh, nw), name='X')
    K = tvm.placeholder((c, 1, kh, kw), name='K')
    PaddedX = d2ltvm.padding(X, ph, pw) if ph * pw != 0 else X
    # pack X and K
    assert c % tc == 0
    PackedX = tvm.compute(
        (c//tc, nh+ph*2, nw+pw*2, tc),
        lambda c_out, x, y, c_in: PaddedX[c_out*tc + c_in, x, y],
        name='PackedX')
    PackedK = tvm.compute(
        (c//tc, 1, kh, kw, 1, tc),
        lambda c_out, _, x, y, __, c_in: K[
            c_out*tc + c_in, 0, x, y],
        name='PackedK')
    return X, K, PaddedX, PackedX, PackedK
```

```{.python .input  n=5}
c, n, k, tc = 128, 28, 3, 2
X, K, PaddedX, PackedX, PackedK = depthwise_conv_pack(c, n, n, 3, 3, 1, 1, tc)
print(X.shape, K.shape, PaddedX.shape, PackedX.shape, PackedK.shape)
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[128, 28, 28] [128, 1, 3, 3] [128, 30, 30] [64, 30, 30, 2] [64, 1, 3, 3, 1, 2]\n"
 }
]
```

Verify the results by re-implementing the previous example.

## Computation

Since we changed the data layout, we need to re-implement the convolution computation accordingly.

```{.python .input  n=6}
def depthwise_conv(c, nh, nw, kh, kw, ph, pw, sh, sw, tc):
    """2-D conv

    c : number of channels for both input and output.
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding
    sh, sw : height and width strides
    tc : the tiling sizes of channels
    """
    X, K, PaddedX, PackedX, PackedK = depthwise_conv_pack(
        c, nh, nw, kh, kw, ph, pw, tc)
    # reduction axes
    rkh = tvm.reduce_axis((0, kh), name='rkh')
    rkw = tvm.reduce_axis((0, kw), name='rkw')
    # output height and weights
    oh = d2ltvm.conv_out_size(nh, kh, ph, sh)
    ow = d2ltvm.conv_out_size(nw, kw, pw, sw)
    # Compuated Y in the packed layout
    PackedY = tvm.compute(
        (c//tc, oh, ow, tc),
        lambda c_out, x, y, c_in: tvm.sum(
            (PackedX[c_out, x*sh+rkh, y*sw+rkw, c_in] *
             PackedK[c_out, 0, rkh, rkw, 0, c_in]),
            axis=[rkh, rkw]), name='PackedY')
    
    # Unpack the result
    Y = tvm.compute((c, oh, ow),
                    lambda c, x, y: PackedY[c//tc, x, y, c%tc],
                    name='Y')
    return X, K, Y, PaddedX, PackedX, PackedK, PackedY
```

Let's compile it using the default scheduling and compute the results.

```{.python .input  n=7}
c, n, k, p, s, tc = 6, 12, 3, 1, 1, 2
X, K, Y, _, _, _, _ = depthwise_conv(c, n, n, k, k, p, p, s, s, tc)
mod = tvm.build(tvm.create_schedule(Y.op), [X, K, Y])

data, weight, out = d2ltvm.get_conv_data(c, c, n, k, p, s, tvm.nd.array, conv_type='depthwise')
mod(data, weight, out)
```

And then verify the result.

```{.python .input  n=11}
data, weight, bias, out_mx = d2ltvm.get_conv_data_mxnet(c, c, n, k, p, s, conv_type='depthwise')
d2ltvm.depthwise_conv_mxnet(data, weight, bias, out_mx, k, p, s)
np.testing.assert_allclose(out_mx[0].asnumpy(), out.asnumpy(), atol=1e-5)
```

## Schedule

The optimization strategy here is similar to :numref:`ch_conv_cpu`. The major differences are

1. the innermost axis is the inner axis split from the output channel because the elements have already sit on the last dimension after packing.
2. We only split the width dimension instead of both width and height dimensions.
3. We need to schedule the packing and unpacking computations as well.

```{.python .input  n=18}
# tiling sizes for output channel, input channel, and width
toc, tic, tw = 16, 16, 4

def cached_block(oc, ic, n, k, p, s):
    X, K, Y, PaddedX, PackedX, PackedK, PackedY = conv(
        oc, ic, n, n, k, k, p, p, s, s, toc, tic)
    s = tvm.create_schedule(Y.op)
    CachedY = s.cache_write(PackedY, 'local')
    oc_out, h, w, oc_in = s[PackedY].op.axis
    oc_out_h = s[PackedY].fuse(oc_out, h)
    # Parallel on the first two dimensions oc_out and h
    s[PackedY].parallel(oc_out_h)
    # Optimize the computation of a cached output block
    w_out, w_in = s[PackedY].split(w, factor=tw)  # Split the columns
    s[CachedY].compute_at(s[PackedY], w_out)
    _, _, cw, oc_in = CachedY.op.axis
    ric_out, rkh, rkw, ric_in = CachedY.op.reduce_axis
    s[CachedY].reorder(ric_out, rkh, rkw, ric_in, cw, oc_in)
    s[CachedY].unroll(ric_in)
    s[CachedY].unroll(cw)
    s[CachedY].vectorize(oc_in)
    # Schedule the padding by adding thread-level parallelism
    if PaddedX != X:
        s[PaddedX].parallel(PaddedX.op.axis[0])
    # Optimize the packing of X and K
    s[PackedX].parallel(s[PackedX].fuse(*PackedX.op.axis[0:2]))
    s[PackedX].unroll(PackedX.op.axis[-1])
    s[PackedK].parallel(s[PackedK].fuse(*PackedK.op.axis[0:2]))
    s[PackedK].unroll(PackedK.op.axis[-1])
    # Optimize the unpacking of Y
    s[Y].parallel(s[Y].fuse(*Y.op.axis[0:2]))
    s[Y].unroll(Y.op.axis[-1])
    return s, (X, K, Y)

s, args = cached_block(32, 32, 64, 3, 1, 1)
# Uncomment the following line to see the long
# psuedo codes because of unrolling.
# tvm.lower(s, args, simple_mode=True)
```

Let's benchmark then same workloads as :numref:`ch_conv_cpu` and compare to our MXNet baseline. As you can see, the results are significantly improved.

```{.python .input  n=19}
channels = 2**np.arange(4, 9)
sizes = [(int(c), 64, 3) for c in channels] # a list of (c, n, k)
tvm_gflops = d2ltvm.bench_conv_tvm(cached_block, sizes, target)
mxnet_gflops = d2ltvm.bench_conv_mxnet(sizes)
d2ltvm.plot_gflops(channels, [mxnet_gflops, tvm_gflops], ['mxnet', 'tvm'])
```

## Summary

- We often tile input and output channels to for better cache efficiency and pack data accordingly.

## Exercises

- What if the number of channels cannot be divided by the tiling size?
- Try different tiling sizes.
