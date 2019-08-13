# Vector Addition
:label:`ch_vector_add`

Let's begin with a simple example: summing two vectors. It's straightforward in NumPy.

```{.python .input  n=13}
import numpy as np

n = 100
a = np.random.normal(size=n).astype(np.float32)
b = np.random.normal(size=n).astype(np.float32)
c = a + b
```

Here we create two random vectors with length 100, and sum them element-wisely. Note that NumPy in default use 64-bit floating-points or 64-bit integers. We often use 32-bit floating point in deep learning, so we explicitly cast the data type.

Despite that we can use the build-in `+` operator in NumPy, let's try to implement it by only use scalar operators. It will help understand the implementation with TVM. The following function uses a for-loop to iterate every element, and then sum two elements with the scalar `+` operator.

```{.python .input  n=14}
def vector_add(a, b, c):
    for i in range(n):
        c[i] = a[i] + b[i]

d = np.empty(shape=n, dtype=np.float32)
vector_add(a, b, d)
np.testing.assert_array_equal(c, d)
```

## Define the Computation

Now let's implement `vector_add` in TVM. The TVM implementation differs above in two ways:

1. We don't need to write the complete function, but only to specify how each element of the output, i.e. `c[i]`, is computed
1. TVM is symbolic, we create symbolic variable by specifying its shape, and define how the program will be computed

In the following program, we first declare the placeholders `A` and `B` for both inputs by specifying their shapes, `(n,)`, through `tvm.placeholder`. Both `A` and `B` are `Tensor` objects, we can feed with data later. We assign names to them so we can print an easy-to-read program later.

Next we define how the output `C` is computed by `tvm.compute`. It accepts two arguments, the output shape, and a function to compute each element by giving its index. Since the output is a vector, its element by be indexed by a single integer. There the lambda function accepts a single argument `i`. The computation is similar to the `vector_add` function we defined before except that TVM will fill in the for loop later.

```{.python .input  n=17}
import tvm

A = tvm.placeholder((n,), name='a')
B = tvm.placeholder((n,), name='b')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='c')

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

Once both computation and schedule are defined, we can compile them into an executable module with `tvm.build`. We can specify various targets such as GPU. Here we just use the default CPU target.

```{.python .input}
tvm_vector_add = tvm.build(s, [A, B, C])
```

The compiled module accepts three arguments, `A`, `B` and `C`. All of them need to be a `tvm.ndarray.NDArray` object. The easiest way is creating a NumPy ndarray and then convert into TVM ndarray by `tvm.nd.array`. We can convert it back to NumPy by the `asnumpy` method.

```{.python .input}
x = np.ones(2)
y = tvm.nd.array(x)
type(y), y.asnumpy()
```

We define a convenient function to evaluate a module by automatically converting NumPy ndarrays arguments into TVM ndarrays, and then return results in NumPy formats. This function is saved in the `d2ltvm` package so we can reuse it later.

```{.python .input  n=82}
# Save to the d2ltvm package.
def eval_mod(mod, *args):
    tvm_args = [tvm.nd.array(arr) for arr in args]
    mod(*tvm_args)
    return [arr.asnumpy() for arr in tvm_args]
```

Now evaluate and check the results.

```{.python .input}
c = np.empty(shape=n, dtype=np.float32)
_, _, e = eval_mod(tvm_vector_add, a, b, c)
np.testing.assert_array_equal(e, d)
```

## Argument Constraints

Remember that we specified both inputs should be 100-length vectors when declaring `A` and `B`.

```{.python .input}
A.shape, B.shape, C.shape
```

TVM will check if the input shapes satisfy this specification.

```{.python .input  n=81}
try:
    a, b, c = (np.ones(101, dtype='float32') for _ in range(3))
    eval_mod(tvm_vector_add, a, b, c)
except tvm.TVMError as e:
    print(e)
```

The default data type in TVM in `float32`.

```{.python .input}
A.dtype, B.dtype, C.dtype
```

An error will appear if input with a different data type.

```{.python .input}
try:
    a, b, c = (np.ones(100, dtype='int32') for _ in range(3))
    eval_mod(tvm_vector_add, a, b, c)
except tvm.TVMError as e:
    print(e)
```

## Summary

Implementing an operator using TVM has three steps:

1. Declare the computation by specifying input and output shapes and how each output element is computed.
2. Create a schedule to fully utilize the machine resources.
3. Compile to the hardware target.
