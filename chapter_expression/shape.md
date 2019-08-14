# Shapes

We implemented a vector addition in :numref:`ch_vector_add` to sum two 100-length vectors. It's too restrictive for real scenarios that inputs can have arbitrary shapes. In this section, we will show how to deal with arbitrary shape inputs.

## Variable Shapes

Remember that we create symbolic placeholders for tensors `A` and `B` so we can feed with data later. We can do the same thing for the shape as well. In particular, we use `tvm.var` to create a symbolic variable for a `int32` scalar.

```{.python .input  n=1}
import d2ltvm as d2l
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

Compared to the generated pseudo codes in :numref:`ch_vector_add`, we can see the upper value of the for loop is changed from 100 to `n`. 

Next we put the test codes in the `test_mod` function to verify the compiled module is able to correctly execute on input vectors with different lengths.

```{.python .input  n=4}
def test_mod(mod, size):
    a, b = (np.random.normal(size=size).astype('float32') for _ in range(2))
    c = np.empty(size, dtype='float32')
    d2l.eval_mod(mod, a, b, c)
    print('c.shape:', c.shape)
    np.testing.assert_equal(c, a + b)
    

mod = tvm.build(s, [A, B, C])    
test_mod(mod, 5)
test_mod(mod, 1000)
```

But note that we still place the constraint that `A`, `B`, and `C` should be the same shape. So an error will occur if it is not satisfied.

## Multi-dimensional Shapes

You may already note that a shape is presented as a list, a  tuple. A single element tuple means a 1-D tensor, or a vector. We can extend it to multi-dimensional tensors by adding variables to the shape list. 

The following function builds a module for multi-dimensional tensor addition, the number of dimensions is specified by `ndim`. Note that instead of using `lambda i, j: A[i, j] + B[i, j]` for `ndim=2` and `lambda i, j, k: A[i, j, k] + B[i, j, k]` for `ndim=3`, we use `*i` to handle the general multi-dimensional case.

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
test_mod(mod, (2,3,4,5))
```

## Summary

- We can use `tvm.var()` when we don't know the shape beforehand. 
- The shape of a $n$-dimensional tensor is presented by a $n$-length list.
