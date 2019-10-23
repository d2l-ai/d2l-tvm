# Vector Addition
:label:`ch_vector_add`

Now you have installed all libraries, let's write our first program: summing two vectors `a` and `b`. It's straightforward in NumPy, we can do it by `c = a + b`.

```{.python .input  n=2}
import numpy as np

np.random.seed(0)
n = 100
a = np.random.normal(size=n).astype(np.float32)
b = np.random.normal(size=n).astype(np.float32)
c = a + b
```

Here we create two random vectors with length 100, and sum them element-wisely. Note that NumPy in default use 64-bit floating-points or 64-bit integers. We often use 32-bit floating point in deep learning, so we explicitly cast the data type.

Despite that we can use the build-in `+` operator in NumPy, let's try to implement it by only use scalar operators. It will help understand the implementation with TVM. The following function uses a for-loop to iterate every element, and then sum two elements with the scalar `+` operator.

```{.python .input  n=3}
def vector_add(a, b, c):
    for i in range(n):
        c[i] = a[i] + b[i]

d = np.empty(shape=n, dtype=np.float32)
vector_add(a, b, d)
np.testing.assert_array_equal(c, d)
```

Given we will frequently create two random ndarrays and another empty one to store the results in the following chapters, we save this routine to reuse it in the future.

```{.python .input}
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

Note that we fixed the random seed so that we will always get the same results to simplify the comparison between NumPy, TVM and others. In addition, it accepts an optional `constructor` to  convert the data into a different format. 

## Define the Computation
:label:`def_com`

Now let's implement `vector_add` in TVM. The TVM implementation differs above in two ways:

1. We don't need to write the complete function, but only to specify how each element of the output, i.e. `c[i]`, is computed
1. TVM is symbolic, we create symbolic variable by specifying its shape, and define how the program will be computed

In the following program, we first declare the placeholders `A` and `B` for both inputs by specifying their shapes, `(n,)`, through `tvm.placeholder`. Both `A` and `B` are `Tensor` objects, we can feed with data later. We assign names to them so we can print an easy-to-read program later.

Next we define how the output `C` is computed by `tvm.compute`. It accepts two arguments, the output shape, and a function to compute each element by giving its index. Since the output is a vector, its element by be indexed by a single integer. There the lambda function accepts a single argument `i`, and return `c[i]`, which is identical to `c[i] = a[i] + b[i]` defined in `vector_add`. One difference is that we didn't write the for-loop, which will be filled by TVM later.

```{.python .input  n=6}
import tvm

# Save to the d2ltvm package.
def vector_add(n):
    """TVM expression for vector addition"""
    A = tvm.placeholder((n,), name='a')
    B = tvm.placeholder((n,), name='b')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='c')
    return A, B, C

A, B, C = vector_add(n)
type(C)
```

We can see that `C` is again a `Tensor` object. The operator, i.e. how `C` is computed, can be accessed by `C.op`.

## Create a Schedule

To run the computation, we need to specify how to execute the program, for example, the order to access data and how to do multi-threading. Such an execution plan is called a schedule. A schedule often doesn't not change the results, but a good schedule fully utilizes the machine resources to achieve high performance.

Let's create a default schedule on the operator and print the pseudo codes.

```{.python .input  n=16}
s = tvm.create_schedule(C.op)

tvm.lower(s, [A, B, C], simple_mode=True)
```

As can be seen, the pseudo codes are C-like. TVM adds proper for-loops according to the output shape. In overall, it quite similar to the preview function `vector_add`.

An efficient schedule is often closely related to the hardware we are using. We will explore various options in :numref:`ch_schedule` later.

## Compile and Run

Once both computation and schedule are defined, we can compile them into an executable module with `tvm.build`. We must specify two arguments, the schedule, and all and  We can specify various targets such as GPU. Here we just use the default CPU target.

```{.python .input}
tvm_vector_add = tvm.build(s, [A, B, C])
```

The compiled module accepts three arguments, `A`, `B` and `C`. All of them need to be a `tvm.ndarray.NDArray` object. The easiest way is creating a NumPy ndarray and then convert into TVM ndarray by `tvm.nd.array`. We can convert it back to NumPy by the `asnumpy` method.

```{.python .input}
x = np.ones(2)
y = tvm.nd.array(x)
type(y), y.asnumpy()
```

Now let's construct the same `a` and `b`, but returns them as TVM ndarrays.

```{.python .input}
a, b, d = get_abc(100, tvm.nd.array)
```

Do the computation with `tvm_vector_add`, the result `d` should be equal to `c` that is computed through NumPy.

```{.python .input}
tvm_vector_add(a, b, d)
np.testing.assert_array_equal(c, d.asnumpy())
```

## Argument Constraints

Remember that we specified both inputs should be 100-length vectors when declaring `A` and `B`.

```{.python .input}
A.shape, B.shape, C.shape
```

TVM will check if the input shapes satisfy this specification.

```{.python .input  n=81}
try:
    a, b, c = get_abc(200, tvm.nd.array)
    tvm_vector_add(a, b, c)
except tvm.TVMError as e:
    print(e)
```

The default data type in TVM is `float32`.

```{.python .input}
A.dtype, B.dtype, C.dtype
```

An error will appear if input with a different data type.

```{.python .input}
try:
    a, b, c = get_abc(100, tvm.nd.array)
    a = tvm.nd.array(a.asnumpy().astype('float64'))
    tvm_vector_add(a, b, c)
except tvm.TVMError as e:
    print(e)
```

## Summary

Implementing an operator using TVM has three steps:

1. Declare the computation by specifying input and output shapes and how each output element is computed.
2. Create a schedule to fully utilize the machine resources.
3. Compile to the hardware target.
