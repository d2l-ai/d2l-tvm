# Depthwise Convolution
:label:`ch_conv`

Depthwise convolution is a special kind of convolution commonly used in convolutional neural networks designed for mobile and embedding applications, e.g. MobileNet :cite:`Howard.Zhu.Chen.ea.2017`.

```{.python .input  n=1}
import d2ltvm
import numpy as np
import tvm
```

## Compute definition

Let's revisit the 2-D convolution described in :numref:`ch_conv` first. The 2-D convolution basically takes a 3-D data (note that for simplicity we set the batch size to be 1) in size `(ic, ih, iw)`, convolves it with a 4-D kernel in size `(oc, ic, kh, kw)`, and produces an output data in size `(oc, oh, ow)`. During the convolution, some padding and stride may be applied.

For depthwise convolution, the convolution computation itself stays the same, as illustrated in :numref:`fig_conv_strides`. It differs from the normal 2-D convolution in the way of organizing the convolution. In order to generate an output data in size `(oc, oh, ow)` from input data in size `(ic, ih, iw)`, a two-stage computation is needed. First, we process the input data with `ic` kernels, each of which convolves with the corresponding channel, to produce an intermediate data in size `(ic, oh, ow)`; then we perform the normal, but pointwise, 2-D convolution on the intermediate data in size `(ic, oh, ow)` using a 4-D kernel in size `(oc, ic, 1, 1)` to produce the output data in size `(oc, oh, ow)`, where `padding=0` and `stride=1`.

The computation of the second stage has been covered in :numref:`ch_conv`. This section only focuses on the computation of the first stage, which is referred to as depthwise convolution. :numref:`fig_depthwise_conv` illustrates its computation procedure.

![Illustration of a depthwise convolution. Each channel of the input data convolves with a dedicated kernel.](../img/depthwise-conv.svg)
:label:`fig_depthwise_conv`

From the figure we can see that the shape of the weight is a bit different from the 2-D convolution. The weight for depthwise convolution is 3-D, while it is 4-D for 2-D convolution. Therefore, we modify the `get_conv_data` slightly to handle the generation of the data for depthwise convolution, and save it for future use.

```{.python .input  n=8}
# Save to the d2ltvm package.
def get_conv_data(oc, ic, n, k, p=0, s=1, constructor=None, conv_type='direct'):
    """Return random 3-D data tensor, 3-D kernel tenor and empty 3-D output 
    tensor with the shapes specified by input arguments.

    oc, ic : output and input channels
    n : input width and height
    k : kernel width and height
    p : padding size, default 0
    s : stride, default 1
    conv_type: either direct 2D or depthwise, default direct
    constructor : user-defined tensor constructor
    """
    np.random.seed(0)
    data = np.random.normal(size=(ic, n, n)).astype('float32')
    ic_weight = ic
    if conv_type == 'depthwise':
        ic_weight = 1
    weight = np.random.normal(size=(oc, ic_weight, k, k)).astype('float32')
    on = d2ltvm.conv_out_size(n, k, p, s)
    out = np.empty((oc, on, on), dtype='float32')
    if constructor:
        data, weight, out = (constructor(x) for x in [data, weight, out])
    return data, weight, out
```

Comparing to :numref:`ch_conv`, we added one argument to describe the convolution type, and make the input channel of the weight to be 1 when it is a depthwise convolution. You may wonder why we choose this dimension. The reason is to match the convention brought by the framework.

Then we define the depthwise convolution via TVM. Here, we reuse the `padding` and `conv_out_size` methods defined in :numref:`ch_conv`.

```{.python .input  n=56}
from d2ltvm import padding, conv_out_size

# Save to the d2ltvm package.
def depthwise_conv(ic, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
    """Convolution

    ic : number of channels for both input and output
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding sizes, default 0
    sh, sw : height and width strides, default 1
    """
    # reduction axes
    rkh = tvm.reduce_axis((0, kh), name='rkh')
    rkw = tvm.reduce_axis((0, kw), name='rkw')
    # output height and weights
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    # pad X and then compute Y
    X = tvm.placeholder((ic, nh, nw), name='X')
    K = tvm.placeholder((ic, 1, kh, kw), name='K')
    PaddedX = padding(X, ph, pw) if ph * pw != 0 else X
    Y = tvm.compute(
        (ic, oh, ow),
        lambda c, i, j: tvm.sum(
            (PaddedX[c, i*sh+rkh, j*sw+rkw] * K[c, 0, rkh, rkw]),
            axis=[rkh, rkw]), name='Y')
    
    return X, K, Y, PaddedX
```

After defining the computation of depthwise convolution, we can use the default schedule to compile and execute it as follows.
We also print out the pseudo-code of it.

```{.python .input}
ic, n, k, p, s = 256, 12, 3, 1, 1

X, K, Y, _ = depthwise_conv(ic, n, n, k, k, p, p, s, s)
sch = tvm.create_schedule(Y.op)
mod = tvm.build(sch, [X, K, Y])
print(tvm.lower(sch, [X, K, Y], simple_mode=True))

data, weight, out = get_conv_data(ic, ic, n, k, p, s, 
                                  constructor=tvm.nd.array, 
                                  conv_type='depthwise')
mod(data, weight, out)
```

## Depthwise Convolution in General

You may wonder why we want to replace a typical 2-D convolution into a more complicated, two-stage depthwise plus pointwise 2-D convolution. This book doesn't discuss about the choice of algorithms, but from the computational perspective, the main reason is to reduce the number of computation it requires. Assuming that the input data is in size `[ic, ih, iw]`, the kernel is in size `[kh, kw]`, and the output data is in size `[oc, oh, ow]`, a 2-D convolution takes $2 \times ic \times oh \times ow \times kh \times kw \times oc$ FLOPs, while a depthwise plus pointwise 2-D convolution takes $2 \times ic \times oh \times ow \times (kh \times kw + oc)$ FLOPs. It is easy to see that the 2-D convolution normally takes more FLOPs than depthwise plus pointwise 2-D convolution, especially when the kernel size and/or the number of output channels are large. Taking the above example where $ic=256, oh=ow=12, kh=kw=3$, if we set $oc=512$, the total FLOPs of a 2-D convolution is $339,738,624$, while the depthwise plus pointwise convolution is $38,412,288$, almost one order of magnitude smaller, are much suitable for mobile and embedded applications.

In the MobileNet paper :cite:`Howard.Zhu.Chen.ea.2017`, the depthwise convolution was described as a separable convolution which separates the channels for convolution. From another aspect, a depthwise convolution can be treated as a special kind of grouped convolution. A `G`-grouped convolution divide the channels into `G` groups and do the convolution group by group independently. This was first introduced in AlexNet to save memory. We can easily figure out that when the number of groups equals the number of channels, a grouped convolution is reduced to a depthwise convolution.

In fact, MXNet uses the same API `mx.nd.Convolution` to process depthwise convolution by specifying the number of groups, as we will show in the next code block.

In addition, a depthwise convolution can be generalized in other ways. For example, we can specify a `multiplier` to increase the number of channels for the output of depthwise convolution, which we are cover in this section for simplicity.

## Comparing to Baseline

We use MXNet’s convolution operator as the ground truth to verify the correctness of our depthwise convolution. Before that, we will need to create data. As for TVM, we modify the `get_conv_data_mxnet` defined in :numref:`ch_conv` to take `conv_type`.

```{.python .input  n=10}
import mxnet as mx

# Save to the d2ltvm package.
def get_conv_data_mxnet(oc, ic, n, k, p, s, ctx='cpu', conv_type='direct'):
    ctx = getattr(mx, ctx)()
    data, weight, out = get_conv_data(oc, ic, n, k, p, s, 
                                      constructor=lambda x: mx.nd.array(x, ctx=ctx),
                                      conv_type=conv_type)
    data, out = data.expand_dims(axis=0), out.expand_dims(axis=0)
    bias = mx.nd.zeros(out.shape[1], ctx=ctx)
    return data, weight, bias, out
```

Then we do the computation and compare with the TVM result.

```{.python .input}
# Save to the d2ltvm package.
def depthwise_conv_mxnet(data, weight, bias, out, k, p, s):
    mx.nd.Convolution(data, weight, bias, kernel=(k,k), stride=(s,s),
                      pad=(p,p), num_filter=out.shape[1], 
                      out=out, num_group=weight.shape[0])

data, weight, bias, out_mx = get_conv_data_mxnet(ic, ic, n, k, p, s, conv_type='depthwise')
depthwise_conv_mxnet(data, weight, bias, out_mx, k, p, s)

np.testing.assert_allclose(out_mx[0].asnumpy(), out.asnumpy(), atol=1e-5)
```

## Summary

- Depthwise convolution, together with pointwise convolution, can save a lot of computation and memory compared to normal 2-D convolution.
- Depthwise convolution takes kernels in 3-D, while normal 2-D convolution takes kernels in 4-D.
