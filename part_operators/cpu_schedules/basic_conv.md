# Basic Convolution
:label:`ch_basic_conv_cpu`

The convolution operator is the one of the most expensive but also widely used operators in neural networks. In this chapter, we will cover the operator with single input and output channels. Please refer to chapter [6.2](http://numpy.d2l.ai/chapter_convolutional-neural-networks/conv-layer.html) and [6.3](http://numpy.d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html) in D2L for more explanation about this operator.

```{.python .input  n=1}
import tvm
import numpy as np
from mxnet import np as mp, npx as mpx
import d2ltvm
import timeit
import time

target = 'llvm -mcpu=core-avx2'
```

## Definition

Given a $n_h\times n_w$ data matrix $X$, if there are non-zero paddings $p_w > 0, p_h >0$, then we add $p_h$ rows with 0s on top and bottom, and then add $p_w$ columns with 0s on left and right to make it a $h+2p_h \times w+2p_w$ matrix. If the kernel matrix $K$ has a size of $k_h\times k_w$, using a stride $s_h$ for height and $s_w$ for width, the output $Y$ will have a shape

$$ \lfloor (n_h-k_h+2p_h)/s_h+1\rfloor  \times \lfloor (n_w-k_w+2p_w)/s_w+1\rfloor.$$

In addition, we can compute $Y_{x,y}$ by

$$ Y_{x,y} = \sum_{i=1}^{k_w}\sum_{j=1}^{k_h} X_{x s_w+i, y s_h+j} K_{i,j}$$

An example is illustrated in :numref:`fig_conv_strides`.

![The 2D convolution with paddings for 2, and strides of 3 and 2 for height and width respectively. The shaded portions are the output element and the input and core array elements used in its computation: $0\times0+0\times1+1\times2+2\times3=8$, $0\times0+6\times1+0\times2+0\times3=6$. ](../img/conv-stride.svg)
:label:`fig_conv_strides`

Next, let's first define a function to calculate the output width or height.

```{.python .input  n=2}
# Save to the d2ltvm package.
def conv_out_size(n, k, p, s):
    """Compute the output size by given input size n, 
    kernel size k, padding p, and stride s"""
    return (n - k + 2 * p)//s + 1
```

To compute an element in the output, we need $2k_hk_w$ floating-point operations. The following function returns the giga floating-point operations needed to compute an convolution, where both input matrix and kernel matrix are assumed to be square for simplicity. Also note that it accepts more than 1 input and output channels, which will be discussed in the next chapter.

```{.python .input  n=3}
# Save to the d2ltvm package.
def conv_gflop(oc, ic, n, k, p, s):
    """Compute the #floating points in a convolution by given
    output channels oc, input channels ic, input size n, kernela size k, 
    padding p and stride s
    """
    on = conv_out_size(n, k, p, s)
    return 2 * oc * ic * on * on * k * k / 1e9
```

## Computation 

Now we implement the basic convolution operator. We first pad the input matrix with 0s if necessary, and then compute the output.

```{.python .input  n=4}
def conv(nh, nw, kh, kw, ph, pw, sh, sw):
    """2-D conv with single input and output channels
    
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding
    sh, sw : height and width strides
    """
    X = tvm.placeholder((nh, nw), name='X')
    K = tvm.placeholder((kh, kw), name='K')
    # Pad rows and columns with 0s 
    if ph != 0 or pw != 0: 
        PaddedX = tvm.compute(
            (nh+ph*2, nw+pw*2), 
            lambda x, y: tvm.if_then_else(
                tvm.any(x<ph, x>=nh+ph, y<pw, y>=nw+pw), 0, X[x-ph, y-pw]),
            name='PaddedX')
    else:
        PaddedX = X
    # Reduction axies to iterate the kernel
    rkh = tvm.reduce_axis((0, kh), name='rkh')
    rkw = tvm.reduce_axis((0, kw), name='rkw')
    # Output height and width
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    Y = tvm.compute(
        (oh, ow), 
        lambda x, y: tvm.sum(
            PaddedX[x*sh+rkh, y*sw+rkw] * K[rkh, rkw], axis=[rkh, rkw]), 
        name='Y')
    return X, K, Y, PaddedX
```

A convenient function to create an ndarray. (TODO, move to an early chapter)

```{.python .input  n=5}
# Save to the d2ltvm package.
def get_data(shape, func, target=tvm.cpu()):
    if func.__name__ in ('normal', 'uniform'):
        data = func(size=shape)
    else:
        data = func(shape=shape)
    if hasattr(data, 'asnumpy'):
        data = data.asnumpy()
    return tvm.nd.array(data.astype('float32'), target)
```

Let's see if the padding works as expected.

```{.python .input  n=6}
n, k, p, s = 4, 3, 1, 1
X, _, _, PaddedX = conv(n, n, k, k, p, p, s, s)
mod = tvm.build(tvm.create_schedule(PaddedX.op), [X, PaddedX])
x, padded_x = get_data(X.shape, mp.ones), get_data(PaddedX.shape, mp.empty)
mod(x, padded_x)
print(padded_x)
```

Before optimizing the performance, we should verity the correctness. The following two functions returns the ndarrays for both MXNet and TVM, where MXNet is used as the ground truth.

```{.python .input  n=7}
# Save to the d2ltvm package.
def get_conv_data_mxnet(oc, ic, n, k, p, s):
    mpx.random.seed(0)
    data = mp.random.normal(size=(1, ic, n, n))
    weight = mp.random.normal(size=(oc, ic, k, k))
    bias = mp.zeros((oc,))
    on = conv_out_size(n, k, p, s)
    out = mp.empty((1, oc, on, on))
    # Wait data are generated to make later benchmarking accurate
    mpx.waitall()  
    return data, weight, bias, out

def get_conv_data_tvm(n, k, p, s):
    mpx.random.seed(0)
    on = conv_out_size(n, k, p, s)
    data = d2ltvm.get_data((n, n), mp.random.normal)
    weight = d2ltvm.get_data((k, k), mp.random.normal)
    out = d2ltvm.get_data((on, on), mp.empty)
    return data, weight, out
```

Then test multiple configurations using the default schedule.

```{.python .input}
def test_conv(schedule_generator):
    for n, k, p, s in ((64, 7, 0, 1), (55, 4, 1, 2), (35, 5, 3, 3)):
        # MXNet results
        data, weight, bias, out = get_conv_data_mxnet(1, 1, n, k, p, s)
        mpx.convolution(
            data, weight, bias, kernel=(k,k), pad=(p,p), 
            stride=(s,s), num_filter=1, out=out)
        out = out.squeeze(axis=(0,1))
        # TVM results
        data_tvm, weight_tvm, out_tvm = get_conv_data_tvm(n, k, p, s)
        args = conv(n, n, k, k, p, p, s, s)        
        mod = tvm.build(schedule_generator(*args), args[:3], target)
        mod(data_tvm, weight_tvm, out_tvm)
        # Compare results
        np.testing.assert_allclose(out.asnumpy(), out_tvm.asnumpy())

test_conv(lambda *args: tvm.create_schedule(args[2].op))
```

## Schedule

We introduced block tiling with a write cache in :numref:`ch_block_matmul_cpu`, which can be also used here. But will only split the columns for simplicity. Also we parallelize the padding computation.

```{.python .input  n=8}
tw = 16  # A good tile size 

def conv_schedule(X, K, Y, PaddedX):
    s = tvm.create_schedule(Y.op)
    # Compute output rows in parallel
    h, w = Y.op.axis  # Use h/w instead of x/y to avoid confusion with X/Y.
    s[Y].parallel(h)
    # Optimize the computation of a cached line
    CachedY = s.cache_write(Y, 'local')
    wo, wi = s[Y].split(w, factor=tw) # split the columns
    s[CachedY].compute_at(s[Y], wo)            
    _, cw = CachedY.op.axis
    rkh, rkw = CachedY.op.reduce_axis
    s[CachedY].reorder(rkh, rkw, cw)
    s[CachedY].vectorize(cw)
    # Opimize the padding
    if PaddedX != X:
        s[PaddedX].parallel(PaddedX.op.axis[0])
    # Uncomment it to print the c-like program
    # print(tvm.lower(s, [X, K, Y], simple_mode=True))
    return s
```

Note that little can be optimized for the reduction axes `rkh` and `rkw` because of their sequential nature. Therefore we move the inner axis split from columns as the innermost axis so we can vectorize its computation. 

Test the correctness of this schedule.

```{.python .input  n=9}
test_conv(conv_schedule)
```

A function to evaluate a complied module in TVM. (TODO, move to an early chapter)

```{.python .input  n=10}
# Save to the d2ltvm package.
def benchmark_mod_tvm(mod, args, target):
    # Estimate the #repeat to run for 1 second, with at least 5 runs
    start = time.time()
    mod(*args)
    nrepeat = int(max(1.0/(time.time() - start), 5))
    ctx = tvm.context(target, 0)
    timer = mod.time_evaluator(mod.entry_name, ctx=ctx, number=nrepeat)
    return timer(*args).mean
```

Here is the function to evaluate the GFLOPS of the schedule we specified. We assume height and width are symmetric for simplicity.

```{.python .input  n=11}
def benchmark_conv_tvm(n, k, p, s):
    data = get_conv_data_tvm(n, k, p, s)
    args = conv(n, n, k, k, p, p, s, s)
    mod = tvm.build(conv_schedule(*args), args[:3], target=target)
    time = benchmark_mod_tvm(mod, data, target)
    return conv_gflop(1, 1, n, k, p, s) / time

n, k, p, s = 128, 7, 3, 1
benchmark_conv_tvm(n, k, p, s)
```

Let's benchmark the performance on various input sizes.

```{.python .input  n=15}
sizes = 2**np.arange(5, 11, 1).astype('int')
tvm_gflops = [benchmark_conv_tvm(int(n), k, p, s) for n in sizes]
```

Also define the benchmark function for MXNet and evaluate its performance on the same sizes.

```{.python .input  n=19}
# Save to the d2ltvm package
def conv_timer_mxnet(oc, ic, n, k, p, s):
    timer = timeit.Timer(
        setup='import d2ltvm\n'
        'from mxnet import npx\n'
        'oc, ic, n, k, p, s = %d, %d, %d, %d, %d, %d\n'
        'data, weight, bias, out = d2ltvm.get_conv_data_mxnet(\n'
        '    oc, ic, n, k, p, s)'%(oc, ic, n, k, p, s),
        stmt='npx.convolution(data, weight, bias, kernel=(k,k), pad=(p,p),\n' 
        '    stride=(s,s), num_filter=oc, out=out); out.wait_to_read()\n')
    # Estimate the #repeat to run for 1 second, with at least 5 runs
    nrepeat = max(int(1.0/timer.timeit(1)), 3)
    time = timer.timeit(nrepeat)
    return conv_gflop(oc, ic, n, k, p, s) / time  * nrepeat

mxnet_gflops = [benchmark_conv_mxnet(1, 1, int(n), k, p, s) for n in sizes]
```

We can see that our optimized convolution is 10x faster than MXNet. The reason is that MXNet, whose backend is based on MKL-DNN, doesn't optimize for the single input/output channel cases, which are rarely used in practice.

```{.python .input}
d2ltvm.plot_gflops(sizes, [mxnet_gflops, tvm_gflops], ['mxnet', 'tvm'])
```

## Summary

- Convolution with single channels is similar to matrix multiplication, we can borrow the optimization ideas before. 

## Exercises

- Vary the tile size.
- Try the block tiling introduces in :numref:`ch_block_matmul_cpu`
