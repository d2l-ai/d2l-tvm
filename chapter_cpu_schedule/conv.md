# Convolution 
:label:`ch_conv_cpu`

In this chapter, we will extend the basic convolution introduced in :numref:`ch_basic_conv` to allow multiple input and output channels. 

```{.python .input  n=1}
import tvm
import d2ltvm
import numpy as np
from mxnet import np as mp, npx as mpx

target = 'llvm -mcpu=core-avx2'
```

## Definition


Assume we have a $c_i \times h \times w $ input matrix $X$, and a $c_o\times c_i\times k_h\times k_w$ kernel weight $K$, here $c_i$ and $c_o$ are the number of input channels and the number of the output channels, respectively. Then the output $Y$ has a shape

$$ c_o \times \lfloor (h-k_h+2p_h)/s_h+1\rfloor  \times \lfloor (w-k_w+2p_w)/s_w+1\rfloor.$$

In particular, the $i$-th submatrix $Y_i$, $i=1,\ldots,c_o$, is computed by 

$$ Y_i = \sum_{j=1}^n X_j \star K_{i,j},$$

where $\star$ is the convolution for single input and output channels defined in :numref:`ch_basic_conv`.

## Data Packing 

Before we jump into the computation, we first define how to pack the input data and kernel weight into a more access efficient layout. The idea is splitting the channel dimension, and then move the inner one as the last dimension. Implement it in NumPy is straightforward. 

```{.python .input}
c, n, tc = 4, 2, 2  # channel, height/width, and tiling size
x = mp.arange(c*n*n).reshape((c, n, n))
print('input:', x.shape, '\n', x)
y = x.reshape(c//tc, n, n, tc).transpose(0, 2, 3, 1)
print('packed:', y.shape, '\n', y)
```

Let's implement the computation, in which we split and transpose both input and output channels.

```{.python .input  n=2}
def conv_pack(oc, ic, nh, nw, kh, kw, ph, pw, toc, tic):
    """Pack data and weight for 2D conv
    
    oc, ic : output and input channels.
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding
    toc, tic : the tiling sizes of the output and input channels
    """
    X = tvm.placeholder((ic, nh, nw), name='X')
    K = tvm.placeholder((oc, ic, kh, kw), name='K')
    if ph != 0 or pw != 0: 
        PaddedX = tvm.compute(
            (ic, nh+ph*2, nw+pw*2), 
            lambda ic, x, y: tvm.if_then_else(
                tvm.any(x < ph, x >= nh+ph, y < pw, y >= nw+pw), 
                0, X[ic, x-ph, y-pw]), 
            name='PaddedX')
    else:
        PaddedX = X
    # pack X and K
    assert ic % tic == 0 and oc % toc == 0
    PackedX = tvm.compute(
        (ic//tic, nh+ph*2, nw+pw*2, tic),
        lambda ic_out, x, y, ic_in: PaddedX[ic_out*tic + ic_in, x, y],
        name='PackedX')
    PackedK = tvm.compute(
        (oc//toc, ic//tic, nh, nw, tic, toc), 
        lambda oc_out, ic_out, x, y, ic_in, oc_in: K[
            oc_out*toc + oc_in, ic_out*tic + ic_in, x, y],
        name='PackedK')
    return X, K, PaddedX, PackedX, PackedK
```

Verify the correctness. 

```{.python .input  n=3}
X, K, _, PackedX, PackedK = conv_pack(c, c, n, n, 1, 1, 0, 0, tc, tc)
mod = tvm.build(tvm.create_schedule(PackedX.op), [X, PackedX])
x = tvm.nd.array(x.asnumpy())
packed_x = d2ltvm.get_data((c//tc, n, n, tc), mp.empty)
mod(x, packed_x)
print('packed:', packed_x.shape, '\n', packed_x)
```

## Computation

Now we can implement the computation. Except that we added channel dimensions compared to the `conv` defined in :numref:`ch_basic_conv_cpu`, we pack both input data and kernel weight to compute the output, and then return the unpacked output. 

```{.python .input  n=4}
def conv(oc, ic, nh, nw, kh, kw, ph, pw, sh, sw, toc, tic, tw):
    """2-D conv
    
    oc, ic : output and input channels.
    h, w : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding
    sh, sw : height and width strides
    toc, tic, tw : the tiling sizes of output channel, input channel and width
    """
    X, K, PaddedX, PackedX, PackedK = conv_pack(
        oc, ic, nh, nw, kh, kw, ph, pw, toc, tic)
    # reduction axes
    ric_in = tvm.reduce_axis((0, tic), name='ric_in')
    ric_out = tvm.reduce_axis((0, ic//tic), name='ric_out')
    rkh = tvm.reduce_axis((0, kh), name='rkh')
    rkw = tvm.reduce_axis((0, kw), name='rkw')
    # output height and weights
    oh = d2ltvm.conv_out_size(nh, kh, ph, sh)
    ow = d2ltvm.conv_out_size(nw, kw, pw, sw)
    PackedY = tvm.compute(
        (oc//toc, oh, ow, toc),
        lambda oc_out, x, y, oc_in: tvm.sum(
            PackedX[ric_out, x*sh+rkh, y*sw+rkw, ric_in] * 
            PackedK[oc_out, ric_out, rkh, rkw, ric_in, oc_in], 
            axis=[ric_out, rkh, rkw, ric_in]), name='Y')
    # unpack the result
    Y = tvm.compute((oc, oh, ow), 
                    lambda oc, x, y: PackedY[oc//toc, x, y, oc%toc], 
                    name='Y')
    return X, K, Y, PaddedX, PackedX, PackedK, PackedY
```

Again, define a function to return the ndarrays

```{.python .input  n=5}
def get_conv_data_tvm(oc, ic, n, k, p, s):
    mpx.random.seed(0)
    on = d2ltvm.conv_out_size(n, k, p, s)
    data = d2ltvm.get_data((ic, n, n), mp.random.normal)
    weight = d2ltvm.get_data((oc, ic, k, k), mp.random.normal)
    out = d2ltvm.get_data((oc, on, on), mp.empty)
    return data, weight, out


```

And verity the correctness using the default schedule. 

```{.python .input}
def test_conv(schedule_generator):
    toc = tic = tw = 4  
    for oc, ic, n, k, p, s in ((8, 16, 64, 7, 0, 1), 
                               (4, 16, 55, 4, 0, 2), 
                               (8, 12, 35, 5, 3, 3)):
        # Compute with MXNet
        data, weight, bias, out = d2ltvm.get_conv_data_mxnet(
            oc, ic, n, k, p, s)
        out = mpx.convolution(data, weight, bias, kernel=(k,k), pad=(p,p), 
                              stride=(s,s), num_filter=oc, out=out)
        out = out.squeeze(axis=0)
        # Compute with TVM
        data_tvm, weight_tvm, out_tvm = get_conv_data_tvm(oc, ic, n, k, p, s)
        args = conv(oc, ic, n, n, k, k, p, p, s, s, toc, tic, tw)
        mod = tvm.build(schedule_generator(*args), args[:3], target)
        mod(data_tvm, weight_tvm, out_tvm)
        # Compare results
        np.testing.assert_allclose(out.asnumpy(), out_tvm.asnumpy(), rtol=1e-2)

test_conv(lambda *args: tvm.create_schedule(args[2].op))
```

## Schedule

The optimization strategy here is similar to :numref:`ch_basic_conv` with the following difference: 1) the innermost axis is the inner axis split from the output channel; 2) We fuse the outer output channel with the height to parallel the workload, 3) packing and unpacking needs optimization as well.

```{.python .input  n=6}
toc, tic, tw = 16, 16, 4  # good tiling sizes

def conv_schedule(X, K, Y, PaddedX, PackedX, PackedK, PackedY):
    # Parallel on the first two dimensions oc_out and h
    s = tvm.create_schedule(Y.op)
    CachedY = s.cache_write(PackedY, 'local')
    oc_out, h, w, oc_in = s[PackedY].op.axis
    oc_out_h = s[PackedY].fuse(oc_out, h)
    s[PackedY].parallel(oc_out_h)
    # Optimzie the computation of a cached output block 
    w_out, w_in = s[PackedY].split(w, factor=tw)  # Split the columns
    s[CachedY].compute_at(s[PackedY], w_out)
    _, _, cw, oc_in = CachedY.op.axis
    ric_out, rkh, rkw, ric_in = CachedY.op.reduce_axis
    s[CachedY].reorder(ric_out, rkh, rkw, ric_in, cw, oc_in)
    s[CachedY].unroll(ric_in)
    s[CachedY].unroll(cw)
    s[CachedY].vectorize(oc_in)
    # Opimize the padding
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
    # Uncomment it to print the c-like program
    # print(tvm.lower(s, [X, K, Y], simple_mode=True))
    return s
```

Verify the correctness. 

```{.python .input  n=7}
test_conv(conv_schedule)
```

Define the benchmark function and evaluate on multiple channel values. 

```{.python .input  n=8}
def benchmark_conv_tvm(oc, ic, n, k, p, s):
    data = get_conv_data_tvm(oc, ic, n, k, p, s)
    args = conv(oc, ic, n, n, k, k, p, p, s, s, toc, tic, tw)
    mod = tvm.build(conv_schedule(*args), args[:3], target=target)
    time = d2ltvm.benchmark_mod_tvm(mod, data, target)
    return d2ltvm.conv_gflop(oc, ic, n, k, p, s) / time

n, k, p, s = 64, 3, 1, 1
sizes = 2**np.arange(5, 10, 1).astype('int')
tvm_gflops = [benchmark_conv_tvm(int(c), int(c), n, k, p, s) for c in sizes]
```

Also collect performance values from MXNet. 

```{.python .input  n=16}
mxnet_gflops = [d2ltvm.benchmark_conv_mxnet(int(c), int(c), n, k, p, s)
                for c in sizes]
d2ltvm.plot_gflops(sizes, [mxnet_gflops, tvm_gflops], ['mxnet', 'tvm'])
```

As can be seen, our optimized version work well for small workloads, but its performance decreases when data cannot fit into the L3 cache, which needs better cache mechanism. 

## Summary

- We often split input and output channels to for better vectorization and pack data accordingly. 

## Exercises

- What if the number of channels cannot be divided by the tiling size.
- Try different tiling sizes.
