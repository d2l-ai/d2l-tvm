# Reduction Operations

Reduction operators often reduce the size of the input tensor, such as `np.sum`. Such operators are often straightforward to implement with for-loops. But it's a little bit more complicated in TVM since we cannot use a Python for-loop directly. In this section, we will describe how to implement these reduction operators.

```{.python .input}
import d2ltvm
import numpy as np
import tvm
```

## Sum

Let's start with summing the rows. In NumPy, we can do it with the `sum` method.

```{.python .input  n=29}
a = np.random.normal(size=(3,4)).astype('float32')
a.sum(axis=1)
```

As we did before, let's implement this function from scratch to help understand the TVM expression.

```{.python .input  n=2}
def sum_rows(a, b):
    n = len(b)
    for i in range(n):
        b[i] = np.sum(a[i,:])

b = np.empty((3,), dtype='float32')
sum_rows(a, b)
b
```

It's fairly straightforward, we first iterate on the first dimension, `axis=0`, and then sum all element on the second dimension to write the results. In NumPy, we can use `:` to slice all elements along that dimension.

Now let's implement the same function in TVM. Compared to the vector addition in :numref:`ch_vector_add`, we used two new functions here. One is `tvm.reduce_axis`, which create an axis for reduction with range from 0 to `m`. It's similar to the `:` used in `sum_rows`, but we need to explicitly specify the range in TVM. The other one is `tvm.sum`, which sum all elements along the reducing axis `k` and returns a scalar.

```{.python .input  n=30}
n, m = tvm.var('n'), tvm.var('m')
A = tvm.placeholder((n, m), name='a')
j = tvm.reduce_axis((0, m), name='j')
B = tvm.compute((n,), lambda i: tvm.sum(A[i, j], axis=j), name='b')
s = tvm.create_schedule(B.op)
tvm.lower(s, [A, B], simple_mode=True)
```

We can see that the generated pseudo codes expand `tvm.sum` into another for loop along axis `k`. As mentioned before, the pseudo codes are C-like, so the index of `a[i,j]` is expanded to `(i*m)+j` with that `a` is a 1-D array.

Now test the results are as expected.

```{.python .input  n=5}
mod = tvm.build(s, [A, B])
c = tvm.nd.array(np.empty((3,), dtype='float32'))
mod(tvm.nd.array(a), c)
np.testing.assert_equal(b, c.asnumpy(), atol=1e-5)
```

We know that `a.sum()` will sum all elements in `a` and returns a scalar. Let's also implement in TVM. To do it, we need another reduction axis along the first dimension, which size is `n`. The result is a scalar, namely a 0-rank tensor, can be created with an empty shape tuple.

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
np.testing.assert_allclose(a.sum(), c.asnumpy())
```

Beyond `tvm.sum`, there are other reduction operators such as `tvm.min` and `tvm.max`. We can also implement customized reduction operators as well.

## Commutative Reduction

An operator $\circ$ is commutative if $a\circ b = b\circ a$. TVM allows to define a customized commutative reduction operator through `tvm.comm_reducer`. It accepts two function arguments, one define how to compute $a\circ b$, the other one specifies the initial value.

Let's use the production by rows, e.g `a.prod(axis=1)`, as an example. Again, we first show how to implement it from scratch.

```{.python .input  n=25}
def prod_rows(a, b):
    n, m = a.shape
    for i in range(n):
        b[i] = 1
        for j in range(m):
            b[i] = b[i] * a[i, j]
```

As can be seen, we need to first initialize the return values to be 1, and then compute the reduction using scalar product `*`. Now define these two functions in TVM, the first one accepts a data type argument to return the initial value of an element in the return. The second one defines $a\circ b$ with two scalar inputs. Then we can create the reduction operator.

```{.python .input}
init = lambda dtype: tvm.const(1, dtype=dtype)
comp = lambda a, b: a * b
product = tvm.comm_reducer(comp, init)
```

The usage of `product` is similar to `tvm.sum`.

```{.python .input  n=26}
n = tvm.var('n')
m = tvm.var('m')
A = tvm.placeholder((n, m), name='a')
k = tvm.reduce_axis((0, m), name='k')
B = tvm.compute((n,), lambda i: product(A[i, k], axis=k), name='b')
s = tvm.create_schedule(B.op)
tvm.lower(s, [A, B], simple_mode=True)
```

The generated pseudo codes are similar to the one for summing by rows, except for the value initialization and the reduction arithmetic.

Again, let's verify the results.

```{.python .input  n=28}
mod = tvm.build(s, [A, B])
b = tvm.nd.array(np.empty((3,), dtype='float32'))
mod(tvm.nd.array(a), b)
np.testing.assert_allclose(a.prod(axis=1), b.asnumpy())
```

## Summary

- We can apply a reduction operator, e.g. `tvm.sum` over a reduction axis `tvm.reduce_axis`.
- We can implement customized commutative reductions by `tvm.comm_reducer`
