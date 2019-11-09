# Shapes

The vector addition module defined in :numref:`ch_vector_add` only accepts vectors with 100-length. It's too restrictive for real scenarios where inputs can have arbitrary shapes. In this section, we will show how to relax this constraint to deal with general cases.

## Variable Shapes

Remember that we create symbolic placeholders for tensors `A` and `B` so we can feed with data later. We can do the same thing for the shape as well. In particular, the follwing code block uses `tvm.var` to create a symbolic variable for an `int32` scalar, whose value can be specified later.

```{.python .input  n=1}
import d2ltvm
import numpy as np
import tvm

n = tvm.var(name='n')
type(n), n.dtype
```

Now we can use `(n,)` to create a placeholder for an arbitrary length vector.

```{.python .input  n=3}
A = tvm.placeholder((n,), name='a')
B = tvm.placeholder((n,), name='b')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='c')
s = tvm.create_schedule(C.op)
tvm.lower(s, [A, B, C], simple_mode=True)
```

Compared to the generated pseudo codes in :numref:`ch_vector_add`, we can see the upper bound value of the for loop is changed from 100 to `n`.

Now we define a similar test function as before to verify that the compiled module is able to correctly execute on input vectors with different lengths.

```{.python .input  n=4}
def test_mod(mod, n):
    a, b, c = d2ltvm.get_abc(n, tvm.nd.array)
    mod(a, b, c)
    print('c.shape:', c.shape)
    np.testing.assert_equal(c.asnumpy(), a.asnumpy() + b.asnumpy())

mod = tvm.build(s, [A, B, C])
test_mod(mod, 5)
test_mod(mod, 1000)
```

But note that we still place the constraint that `A`, `B`, and `C` must be in the same shape. So an error will occur if it is not satisfied.

## Multi-dimensional Shapes

You may already notice that a shape is presented as a tuple. A single element tuple means a 1-D tensor, or a vector. We can extend it to multi-dimensional tensors by adding variables to the shape tuple.

The following method builds a module for multi-dimensional tensor addition, the number of dimensions is specified by `ndim`. For a 2-D tensor, we can access its element by `A[i,j]`, similarly `A[i,j,k]` for 3-D tensors. Note that we use `*i` to handle the general multi-dimensional case in the following code.

```{.python .input  n=5}
def tvm_vector_add(ndim):
    A = tvm.placeholder([tvm.var() for _ in range(ndim)])
    B = tvm.placeholder(A.shape)
    C = tvm.compute(A.shape, lambda *i: A[i] + B[i])
    s = tvm.create_schedule(C.op)
    return tvm.build(s, [A, B, C])
```

Verify that it works beyond vectors.

```{.python .input}
mod = tvm_vector_add(2)
test_mod(mod, (2, 2))

mod = tvm_vector_add(4)
test_mod(mod, (2, 3, 4, 5))
```

## Summary

- We can use `tvm.var()` to specify the dimension(s) of a shape when we don't know the concrete data shape before execution.
- The shape of an $n$-dimensional tensor is presented as an $n$-length tuple.
