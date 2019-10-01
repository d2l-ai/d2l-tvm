# Index and Shape Expressions

We have already seen how to declare a shape as `(n,m)` and access elements sequentially by `a[i,j]`. Both can be variable expressions as well. In this section, we will go through several examples.


## Matrix Transpose

Our first example is matrix transpose `a.T`, in which we access `a`'s elements by columns.

```{.python .input  n=16}
import tvm
import d2ltvm as d2l
import numpy as np

n = tvm.var('n')
m = tvm.var('m')
A = tvm.placeholder((n, m), name='a')
B = tvm.compute((m, n), lambda i, j: A[j, i], 'b')
s = tvm.create_schedule(B.op)
tvm.lower(s, [A, B], simple_mode=True)
```

The computation equals to `b[i,j] = a[j,i]`, but note how TVM expands the 2-D indices to 1-D.

The results should be as expected.

```{.python .input  n=21}
a = np.arange(12, dtype='float32').reshape((3, 4))
b = np.empty((4, 3), dtype='float32')

mod = tvm.build(s, [A, B])
d2l.eval_mod(mod, a, out=b)
b
```

## Reshaping

Next let's use expressions for indexing. The following codes reshape a 2-D array `a` to 1-D by `a.reshape(-1)`. Note that how we convert the 1-D index `i` to the 2-D index `[i/m, i%m]`.

```{.python .input  n=28}
B = tvm.compute((m*n, ), lambda i: A[i/m, i%m], name='B')
s = tvm.create_schedule(B.op)
tvm.lower(s, [A, B], simple_mode=True)
```

Since internally a $n$-D array is treated as a 1-D array, the generated codes simplify the index expression `(i/m)*m + i%m` to `i` to improve the efficiency.

We can implement a general 2-D reshape function as well.

```{.python .input  n=31}
p, q = tvm.var('p'), tvm.var('q')
B = tvm.compute((p, q), lambda i, j: A[(i*q+j)/m, (i*q+j)%m], name='B')
s = tvm.create_schedule(B.op)
tvm.lower(s, [A, B], simple_mode=True)
```

Also note that index expressions are simplified when converting from 2-D to 1-D.

When testing the results, we should be aware that we put no constraint on the output shape, which can have an arbitrary shape `(p, q)`, and therefore TVM will not be able to check $qp = nm$ for us. For example, in the following example we created a `b` with size (20) larger than `a` (12), then only the first 12 elements in `b` are from `a`.

```{.python .input}
mod = tvm.build(s, [A, B])
a = np.arange(12, dtype='float32').reshape((3,4))
b = np.zeros((5, 4), dtype='float32')
d2l.eval_mod(mod, a, out=b)
b
```

## Slicing

Now let's consider a special slicing operator `a[bi::si, bj::si] ` where `bi`, `bj`, `si` and `sj` can be specified later. Now the output shape needs to be computed based on the arguments, and in addition, we need to pass these variables as arguments when compiling the module.

```{.python .input}
bi, bj, si, sj = [tvm.var(name) for name in ['bi', 'bj', 'si', 'sj']]
B = tvm.compute(((n-bi)/si, (m-bj)/sj), lambda i, j: A[i*si+bi, j*sj+bj])
s = tvm.create_schedule(B.op)
# Needs to place variables before tensors, unless
# https://github.com/dmlc/tvm/issues/3793 is resolved.
mod = tvm.build(s, [bi, si, bj, sj, A, B])
```

Now test two cases to verify the correctness.

```{.python .input}
b = np.empty((1, 3), dtype='float32')
d2l.eval_mod(mod, 1, 2, 1, 1, a, out=b)
np.testing.assert_equal(b, a[1::2, 1::1])

b = np.empty((1, 2), dtype='float32')
d2l.eval_mod(mod, 2, 1, 0, 2, a, out=b)
np.testing.assert_equal(b, a[2::1, 0::2])
```

## Summary

- Both shape dimensions and indices can be expressions with variables.
- If a variable doesn't appear in shape alone, e.g. `(..,n,..)`, we need to pass it as an argument.
