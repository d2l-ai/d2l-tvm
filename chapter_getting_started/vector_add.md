# Vector Addition
:label:`ch_vector_add`

Now you have installed all libraries, let's write our first program: summing two `n`-dimensional vectors `a` and `b`. It's straightforward in NumPy, where we can do it by `c = a + b`.

## Implemeting with NumPy

```{.python .input  n=1}
import numpy as np

np.random.seed(0)
n = 100
a = np.random.normal(size=n).astype(np.float32)
b = np.random.normal(size=n).astype(np.float32)
c = a + b
```

Here we create two random vectors with length 100, and sum them element-wisely. Note that NumPy in default uses 64-bit floating-points or 64-bit integers, which is different from 32-bit floating point typically used in deep learning, so we explicitly cast the data type.

Although we can use the build-in `+` operator in NumPy to realize element-wise add, let's try to implement it by only using scalar operators. It will help us understand the implementation with TVM. The following function uses a for-loop to iterate over every element of the vectors, and then add two elements together with the scalar `+` operator each time.

```{.python .input  n=2}
def vector_add(a, b, c):
    for i in range(n):
        c[i] = a[i] + b[i]

d = np.empty(shape=n, dtype=np.float32)
vector_add(a, b, d)
np.testing.assert_array_equal(c, d)
```

Given we will frequently create two random ndarrays and another empty one to store the results in the following chapters, we save this routine to reuse it in the future.

```{.python .input  n=3}
# Save to the d2ltvm package.
def get_abc(shape, constructor=None):
    """Return random a, b and empty c with the same shape.
    """
    np.random.seed(0)
    a = np.random.normal(size=shape).astype(np.float32)
    b = np.random.normal(size=shape).astype(np.float32)
    c = np.empty_like(a)
    if constructor is not None:
        a, b, c = [constructor(x) for x in (a, b, c)]
    return a, b, c
```

Note that we fixed the random seed so that we will always get the same results to facilitate the comparison between NumPy, TVM and others. In addition, it accepts an optional `constructor` to  convert the data into a different format.

## Defining the TVM Computation

Now let's implement `vector_add` in TVM. The TVM implementation differs from above in two ways:

1. We don't need to write the complete function, but only to specify how each element of the output, i.e. `c[i]`, is computed
1. TVM is symbolic, we create symbolic variables by specifying their shapes, and define how the program will be computed

In the following program, we first declare the placeholders `A` and `B` for both inputs by specifying their shapes, `(n,)`, through `tvm.placeholder`. Both `A` and `B` are `Tensor` objects, which we can feed data later. We assign names to them so we can print an easy-to-read program later.

Next we define how the output `C` is computed by `tvm.compute`. It accepts two arguments, the output shape, and a function to compute each element by giving its index. Since the output is a vector, its elements are indexed by integers. The lambda function defined in `tvm.compute` accepts a single argument `i`, and returns `c[i]`, which is identical to `c[i] = a[i] + b[i]` defined in `vector_add`. One difference is that we don't write the for-loop, which will be filled by TVM later.

```{.python .input  n=26}
import tvm

# Save to the d2ltvm package.
def vector_add(n):
    """TVM expression for vector addition"""
    A = tvm.placeholder((n,), name='a')
    B = tvm.placeholder((n,), name='b')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='c')
    return A, B, C

A, B, C = vector_add(n)
type(A), type(C)
```

We can see that `A`, `B`, and `C` are all `Tensor` objects, which can be viewed as a symbolic version of NumPy's ndarray. We can access
the variables' attributes such as data type and shape. But those values don't have concrete values right now.

```{.python .input  n=32}
(A.dtype, A.shape), (C.dtype, C.shape)
```

The operation that generates the tensor object can be accessed by `A.op`.

```{.python .input  n=54}
type(A.op), type(C.op)
```

We can see that the types of the operations for `A` and `C` are different, but they share the same base class `Operation`, which represents an operation that generates a tensor object.

```{.python .input  n=44}
A.op.__class__.__bases__[0]
```

## Creating a Schedule

To run the computation, we need to specify how to execute the program, for example, the order to access data and how to do multi-threading parallization.
Such an execution plan is called a *schedule*. Since `C` is the output tensor, let's create a default schedule on its operator and print the pseudo codes.

```{.python .input  n=48}
s = tvm.create_schedule(C.op)
```

A schedule consists of several stages. Each stage corresponds to an operation to describe how it is scheduled. We can access a particular stage by either `s[C]` or `s[C.op]`.

```{.python .input}
type(s), type(s[C])
```

Later on we will see how to change the execution plan to better utilize the hardware resources to improve its efficiency. Here let's see the default execution plan by printing the C-like pseudo codes.

```{.python .input}
tvm.lower(s, [A, B, C], simple_mode=True)
```

The `lower` method accepts the schedule and input and output tensors. The `simple_mode=True` will print the program in a simple and compact way.
Note that the program has added proper for-loops according to the output shape. Overall, it's quite similar to the preview function `vector_add`.

Now you see that TVM separates the computation and the schedule. The computation defines how the results are computed,
which will not change no matter on what hardware platform you run the program.
On the other hand, an efficient schedule are often hardware dependent, but changing a schedule will not impact the correctness.
The idea of separating computation from schedule is inherited by TVM from Halide :cite:`Ragan-Kelley.Barnes.Adams.ea.2013`.

## Compilation and Execution

Once both computation and schedule are defined, we can compile them into an executable module with `tvm.build`. It accepts the same argument as `tvm.lower`. In fact, it first calls `tvm.lower` to generate the program and then compiles to machine codes.

```{.python .input  n=6}
mod = tvm.build(s, [A, B, C])
type(mod)
```

It returns an executable module object. Now we can feed data for `A`, `B` and `C` to run it. The tensor data must be `tvm.ndarray.NDArray` object. The easiest way is to create NumPy ndarray objects first and then convert them into TVM ndarray by `tvm.nd.array`. We can convert them back to NumPy by the `asnumpy` method.

```{.python .input  n=7}
x = np.ones(2)
y = tvm.nd.array(x)
type(y), y.asnumpy()
```

Now let's construct data and return them as TVM ndarrays.

```{.python .input  n=8}
a, b, c = get_abc(100, tvm.nd.array)
```

Do the computation, and verify  the results.

```{.python .input  n=9}
mod(a, b, c)
np.testing.assert_array_equal(a.asnumpy() + b.asnumpy(), c.asnumpy())
```

## Argument Constraints

Remember that we specified both inputs to be 100-length vectors when declaring `A` and `B`.

```{.python .input  n=10}
A.shape, B.shape, C.shape
```

TVM will check if the input shapes satisfy this specification.

```{.python .input  n=11}
try:
    a, b, c = get_abc(200, tvm.nd.array)
    mod(a, b, c)
except tvm.TVMError as e:
    print(e)
```

The default data type in TVM is `float32`.

```{.python .input  n=12}
A.dtype, B.dtype, C.dtype
```

An error will appear if input with a different data type.

```{.python .input  n=13}
try:
    a, b, c = get_abc(100, tvm.nd.array)
    a = tvm.nd.array(a.asnumpy().astype('float64'))
    mod(a, b, c)
except tvm.TVMError as e:
    print(e)
```

## Saving and Loading a Module

A compiled a module can be saved into disk,

```{.python .input  n=14}
mod_fname = 'vector-add.tar'
mod.export_library(mod_fname)
```

and then loaded back later.

```{.python .input  n=15}
loaded_mod = tvm.module.load(mod_fname)
```

Verify the results.

```{.python .input  n=17}
a, b, c = get_abc(100, tvm.nd.array)
loaded_mod(a, b, c)
np.testing.assert_array_equal(a.asnumpy() + b.asnumpy(), c.asnumpy())
```

## Summary

Implementing an operator using TVM has three steps:

1. Declare the computation by specifying input and output shapes and how each output element is computed.
2. Create a schedule to (hopefully) fully utilize the machine resources.
3. Compile to the hardware target.

In addition, we can save the compiled module into disk so we can load it back later.


## [Discussions](https://discuss.tvm.ai/t/getting-started-vector-addition/4707)
