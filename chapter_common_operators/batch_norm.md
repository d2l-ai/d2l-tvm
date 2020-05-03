# Batch Normalization
:label:`ch_batch_norm`

This section talks about how to use TVM to do batch normalization (`batch_norm`). Like pooling, `batch_norm` is also a common operator in CNN. D2L introduces this operator in [details](https://d2l.ai/chapter_convolutional-modern/batch-norm.html). 

From the calculation perspective, for a given value, `batch_norm` subtracts the $mean$ out of it, and then divide it with the square root of the $variance$, no difference than a regular normalization. It is call `batch_norm` because the mean and variance are attained from the batches of when performed the training. After that, `batch_norm` also applies an affine transformation to the value, i.e. multiplies it with a scale value $gamma$ followed by adding a shift value $beta$. $Gamma$ and $beta$ are attained from the gradient computation of training. Lastly, a small positive value $epsilon$ is added to prevent the divisor to be 0.

In the case of inference, both the mean and variance are determined, so the process of `batch_norm` is just a combination of several simple element-wise operations.

```{.python .input  n=1}
import tvm
from tvm import te
import d2ltvm
import numpy as np
```

## Compute definition

In practice, we are not going to perform `batch_norm` of one value. Instead, the `batch_norm` will be executed on the output of a convolution, namely, 3-D data in (channel, height, weight). Data in different channels have different values of $mean$, $variance$, $gamma$, and $beta$. The calculation can be expressed as the following formula.

$$out[i,:,:] = \frac{data[i,:,:] - mean[i]}{\sqrt{var[i]+\epsilon}} \
* gamma[i] + beta[i] $$

During model training, $mean$ and $var$ are computed from the input $data$. However, in model inference which we focus on here, $mean$ and $var$ are given; therefore we don't need to compute them from $data$.

We will define the compute of this formula. Essentially, `batch_norm` is a combination of a number of simple broadcasting and element-wise calculations. 
Note that in :numref:`ch_bcast_add` we defined a limited `broadcast_add` to perform only broadcast addition for 2-D tensors. If we generalize it to more dimensions and more calculators, we can reuse them to compose the `batch_norm` operator. This is actually what TVM does.

Here, for simplicity, we use TVM basic operators for broadcast calculation. TVM operators are defined in `TOPI`, which stands for Tensor OPerator Inventory. It follows the NumPy convention to override the arithmetic operators (i.e. `+`, `-`, `*`, `/`) for broadcast calculation. The element-wise square root can be found in `TOPI`, too.

The code snippet to define `batch_norm` is as follows.

```{.python .input  n=2}
# Save to the d2ltvm package.
import topi

def batch_norm(c, n, eps=1e-5):
    """batch normalization
    
    c : channels
    N : input width and height
    eps : small positive value to prevent divide 0
    """
    def bcast(A, B, op):
        """Ad-hoc broadcast calculation, broadcasting B to A
        A : te.Tensor
        B : either a te.Tensor or a scalar
        op ：arithmetic operator in string
        """
        # the shapes of two operands of broadcast should be identical
        # or B should be a scalar
        assert isinstance(B, float) or len(A.shape) == len(B.shape)
        if op == '+':
            #f = lambda x, y, z: A[x, y, z] + B[x, 0, 0]
            #C = te.compute(A.shape, f, name='C')
            C = A + B
        elif op == '-':
            C = A - B
        elif op == '*':
            C = A * B
        elif op == '/':
            C = A / B
        else:
            raise ValueError("op should be an arithmetic operator.") 
        return C
        
    X = te.placeholder((c, n, n), name='X')
    Mean = te.placeholder((c, 1, 1), name='Mean')
    Var = te.placeholder((c, 1, 1), name='Var')
    Gamma = te.placeholder((c, 1, 1), name='Gamma')
    Beta = te.placeholder((c, 1, 1), name='Beta')
    C1 = bcast(X, Mean, '-')
    C2 = topi.sqrt(bcast(Var, eps, '+'))
    Y = C1 / C2 * Gamma + Beta
    return X, Mean, Var, Gamma, Beta, Y
```

We can then compile print the IR and compile it. The IR contains several stages but should be easy to follow.

```{.python .input}
c = 32
n = 28
X, Mean, Var, Gamma, Beta, Y = batch_norm(c, n)

sch = te.create_schedule(Y.op)
mod = tvm.build(sch, [X, Mean, Var, Gamma, Beta, Y])

print(tvm.lower(sch, [X, Mean, Var, Gamma, Beta], simple_mode=True))
```

To execute it, we will need to create data for `batch_norm`. Similar to the previous sections for getting conv and pooling data, we define a `get_bn_data` method to generate the data of `batch_norm`. One tricky thing is that the variance must be non-negative numbers. Therefore, we move the mean value of the random number generator's normal distribution to 1 (by default mean 0 and standard deviation 1), and get the absolute numbers of generated results.

After getting the data, we can simply call the compiled module to execute.

```{.python .input  n=3}
# Save to the d2ltvm package.
def get_bn_data(c, n, constructor=None):
    """Return the batch norm data, mean, variance, gamma and beta tensors.
       Also return the empty tensor for output.

    c : channels
    n : input width and height
    constructor : user-defined tensor constructor
    """
    np.random.seed(0)
    data = np.random.normal(size=(c, n, n)).astype('float32')
    mean = np.random.normal(size=(c, 1, 1)).astype('float32')
    # move the mean of the normal distribution to be 1
    var = np.random.normal(loc=1.0, size=(c, 1, 1)).astype('float32')
    # make sure all variance numbers are not negative
    var = np.absolute(var)
    gamma = np.random.normal(size=(c, 1, 1)).astype('float32')
    beta = np.random.normal(size=(c, 1, 1)).astype('float32')
    out = np.empty((c, n, n), dtype='float32')
    if constructor:
        data, mean, var, gamma, beta, out = \
        (constructor(x) for x in [data, mean, var, gamma, beta, out])
    return data, mean, var, gamma, beta, out

data, mean, var, gamma, beta, out = get_bn_data(c, n, tvm.nd.array)
mod(data, mean, var, gamma, beta, out)
```

## MXNet Baseline

We use the `batch_norm` function of MXNet as the baseline to check the correctness of our compiled functions. 
This function in MXNet was defined to be generic for both training and inference. In the inference case that we talk about here, we will need to set the corresponding input arguments properly. 
One is $use_global_stats$, which needs to be set `True` as we will use the input mean and variance for `batch_norm` to compute instead of computing them from the input data (training will do so). 
The other is $fix\_gamma$, which needs to be set `False` so that the input $gamma$ will be used instead of setting $gamma$ to be all 1.

Lastly, like we have discussed in other cases, MXNet `batch_norm` has input data in 4D, including batch as the outmost dimension. So we will expand this dimension in the data accordingly.

```{.python .input  n=7}
import mxnet as mx

# Save to the d2ltvm package.
def get_bn_data_mxnet(c, n, ctx='cpu'):
    ctx = getattr(mx, ctx)()
    data, mean, var, gamma, beta, out = get_bn_data(c, n,
                                      lambda x: mx.nd.array(x, ctx=ctx))
    data, out = data.expand_dims(axis=0), out.expand_dims(axis=0)
    return data, mean, var, gamma, beta, out

# Save to the d2ltvm package.
def batch_norm_mxnet(data, mean, var, gamma, beta, out, eps=1e-5):
    # use_global_stats=True to use the input mean and var instead of computing
    # the mean and var of the input data.
    # fix_gamma=False so that gamma won't be set to 1.
    mx.nd.BatchNorm(data, gamma, beta, mean, var, eps, 
                    use_global_stats=True, fix_gamma=False, out=out)

data, mean, var, gamma, beta, out_mx = get_bn_data_mxnet(c, n)
batch_norm_mxnet(data, mean, var, gamma, beta, out_mx)
```

Finally, we check if our results are close enough to the results produced by MXNet.

```{.python .input  n=9}
np.testing.assert_allclose(out_mx[0].asnumpy(), out.asnumpy(), atol=1e-5)
```

## Summary

- From the computation perspective, `batch_norm` is a combination of a number of broadcast and element-wise simple operators, which can be easily attained from TVM's Tensor OPerator Inventory (TOPI).
- In inference, $mean$ and $var$ of `batch_norm` are pre-defined.
