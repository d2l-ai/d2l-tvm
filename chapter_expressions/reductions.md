# Reduction Operations

Reduction is an operation to reduce certain dimension(s) of an input tensor, usually to scalar(s), e.g. `np.sum` in NumPy. Reduction is often straightforward to implement with for-loops. But it's a little bit more complicated in TVM since we cannot use a Python for-loop directly. In this section, we will describe how to implement reduction in TVM.

```{.python .input}
import d2ltvm
import numpy as np
import tvm
```

## Sum

Let's start with summing the rows of a 2-D matrix to reduce it to be a 1-D vector. In NumPy, we can do it with the `sum` method.

```{.python .input  n=29}
a = np.random.normal(size=(3,4)).astype('float32')
a.sum(axis=1)
```

As we did before, let's implement this operation from scratch to help understand the TVM expression.

```{.python .input  n=2}
def sum_rows(a, b):
    """a is an n-by-m 2-D matrix, b is an n-length 1-D vector 
    """
    n = len(b)
    for i in range(n):
        b[i] = np.sum(a[i,:])

b = np.empty((3,), dtype='float32')
sum_rows(a, b)
b
```

It's fairly straightforward, we first iterate on the first dimension, `axis=0`, and then sum all elements on the second dimension to write the results. In NumPy, we can use `:` to slice all elements along that dimension.

Now let's implement the same thing in TVM. Comparing to the vector addition in :numref:`ch_vector_add`, we used two new operators here. One is `tvm.reduce_axis`, which create an axis for reduction with range from 0 to `m`. It's functionally similar to the `:` used in `sum_rows`, but we need to explicitly specify the range in TVM. The other one is `tvm.sum`, which sums all elements along the reducing axis `k` and returns a scalar.

```{.python .input  n=30}
n, m = tvm.var('n'), tvm.var('m')
A = tvm.placeholder((n, m), name='a')
j = tvm.reduce_axis((0, m), name='j')
B = tvm.compute((n,), lambda i: tvm.sum(A[i, j], axis=j), name='b')
s = tvm.create_schedule(B.op)
tvm.lower(s, [A, B], simple_mode=True)
```

We can see that the generated pseudo codes expand `tvm.sum` into another for loop along axis `k`. As mentioned before, the pseudo codes are C-like, so the index of `a[i,j]` is expanded to `(i*m)+j` by treating `a` as a 1-D array. Also note that `b` is initialized to be all-zero before summation.

Now test the results are as expected.

```{.python .input  n=5}
mod = tvm.build(s, [A, B])
c = tvm.nd.array(np.empty((3,), dtype='float32'))
mod(tvm.nd.array(a), c)
np.testing.assert_equal(b, c.asnumpy())
```

We know that `a.sum()` will sum all elements in `a` and returns a scalar. Let's also implement this in TVM. To do it, we need another reduction axis along the first dimension, whose size is `n`. The result is a scalar, namely a 0-rank tensor, can be created with an empty tuple `()`.

```{.python .input  n=31}
i = tvm.reduce_axis((0, n), name='i')
B = tvm.compute((), lambda: tvm.sum(A[i, j], axis=(i, j)), name='b')
s = tvm.create_schedule(B.op)
tvm.lower(s, [A, B], simple_mode=True)
```

Let's also verify the results.

```{.python .input  n=17}
mod = tvm.build(s, [A, B])
c = tvm.nd.array(np.empty((), dtype='float32'))
mod(tvm.nd.array(a), c)
np.testing.assert_allclose(a.sum(), c.asnumpy(), atol=1e-5)
```

In this case we use `np.testing.assert_allclose` instead of `np.testing.assert_equal` to verify the results as the calculation on `float32` numbers may differ due to the numerical error.

Beyond `tvm.sum`, there are other reduction operators in TVM such as `tvm.min` and `tvm.max`. We can also use them to implement the corresponding reduction operations as well.

## Commutative Reduction

In mathematics, an operator $\circ$ is commutative if $a\circ b = b\circ a$. TVM allows to define a customized commutative reduction operator through `tvm.comm_reducer`. It accepts two function arguments, one defines how to compute $a\circ b$, the other one specifies the initial value.

Let's use the production by rows, e.g `a.prod(axis=1)`, as an example. Again, we first show how to implement it from scratch.

```{.python .input  n=25}
def prod_rows(a, b):
    """a is an n-by-m 2-D matrix, b is an n-length 1-D vector 
    """
    n, m = a.shape
    for i in range(n):
        b[i] = 1
        for j in range(m):
            b[i] = b[i] * a[i, j]
```

As can be seen, we need to first initialize the return values to be 1, and then compute the reduction using scalar product `*`. Now let's define these two functions in TVM to serve as the arguments of `tvm.comm_reducer`. As discussed, the first one defines $a\circ b$ with two scalar inputs. The second one accepts a data type argument to return the initial value of an element. Then we can create the reduction operator.

```{.python .input}
comp = lambda a, b: a * b
init = lambda dtype: tvm.const(1, dtype=dtype)
product = tvm.comm_reducer(comp, init)
```

The usage of `product` is similar to `tvm.sum`. Actually, `tvm.sum` is a pre-defined reduction operator using the same way.

```{.python .input  n=26}
n = tvm.var('n')
m = tvm.var('m')
A = tvm.placeholder((n, m), name='a')
k = tvm.reduce_axis((0, m), name='k')
B = tvm.compute((n,), lambda i: product(A[i, k], axis=k), name='b')
s = tvm.create_schedule(B.op)
tvm.lower(s, [A, B], simple_mode=True)
```

The generated pseudo codes are similar to the one for summing by rows, except for the initialized value and the reduction arithmetic.

Again, let's verify the results.

```{.python .input  n=28}
mod = tvm.build(s, [A, B])
b = tvm.nd.array(np.empty((3,), dtype='float32'))
mod(tvm.nd.array(a), b)
np.testing.assert_allclose(a.prod(axis=1), b.asnumpy(), atol=1e-5)
```

## Summary

- We can apply a reduction operator, e.g. `tvm.sum` over a reduction axis `tvm.reduce_axis`.
- We can implement customized commutative reduction operators by `tvm.comm_reducer`.
