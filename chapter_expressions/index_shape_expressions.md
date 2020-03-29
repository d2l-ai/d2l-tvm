# Index and Shape Expressions

You already know that a shape can be a tuple of symbols such as `(n, m)` and the elements can be accessed via indexing, e.g. `a[i, j]`. In practice, both shapes and indices may be computed through complex expressions. We will go through several examples in this section.

```{.python .input}
import d2ltvm
import numpy as np
import tvm
from tvm import te
```

## Matrix Transpose

Our first example is matrix transpose `a.T`, in which we access `a`'s elements by columns.

```{.python .input  n=16}
n = te.var('n')
m = te.var('m')
A = te.placeholder((n, m), name='a')
B = te.compute((m, n), lambda i, j: A[j, i], 'b')
s = te.create_schedule(B.op)
tvm.lower(s, [A, B], simple_mode=True)
```

Note that the 2-D index, e.g. `b[i,j]` are collapsed to the 1-D index `b[((i*n) + j)]` to follow the C convention.

Now verify the results.

```{.python .input  n=21}
a = np.arange(12, dtype='float32').reshape((3, 4))
b = np.empty((4, 3), dtype='float32')
a, b = tvm.nd.array(a), tvm.nd.array(b)

mod = tvm.build(s, [A, B])
mod(a, b)
print(a)
print(b)
```

## Reshaping

Next let's use expressions for indexing. The following code block reshapes a 2-D array `a` (`n` by `m` as defined above) to 1-D (just like `a.reshape(-1)` in NumPy). Note how we convert the 1-D index `i` to the 2-D index `[i//m, i%m]`.

```{.python .input  n=28}
B = te.compute((m*n, ), lambda i: A[i//m, i%m], name='b')
s = te.create_schedule(B.op)
tvm.lower(s, [A, B], simple_mode=True)
```

Since an $n$-D array is essentially listed in the memory as a 1-D array, the generated code does not rearrange the data sequence, but it simplifies the index expression from 2-D (`(i//m)*m + i%m`) to 1-D (`i`) to improve the efficiency.

We can implement a general 2-D reshape function as well.

```{.python .input  n=31}
p, q = te.var('p'), te.var('q')
B = te.compute((p, q), lambda i, j: A[(i*q+j)//m, (i*q+j)%m], name='b')
s = te.create_schedule(B.op)
tvm.lower(s, [A, B], simple_mode=True)
```

When testing the results, we should be aware that we put no constraint on the output shape, which can have an arbitrary shape `(p, q)`, and therefore TVM will not be able to check if $qp = nm$ for us. For example, in the following example we created a `b` with size (20) larger than `a` (12), then only the first 12 elements in `b` are from `a`, others are uninitialized values.

```{.python .input}
mod = tvm.build(s, [A, B])
a = np.arange(12, dtype='float32').reshape((3,4))
b = np.zeros((5, 4), dtype='float32')
a, b = tvm.nd.array(a), tvm.nd.array(b)

mod(a, b)
print(b)
```

## Slicing

Now let's consider a special slicing operator `a[bi::si, bj::sj] ` where `bi`, `bj`, `si` and `sj` can be specified later. Now the output shape needs to be computed based on the arguments. In addition, we need to pass the variables `bi`, `bj`, `si` and `sj` as arguments when compiling the module.

```{.python .input}
bi, bj, si, sj = [te.var(name) for name in ['bi', 'bj', 'si', 'sj']]
B = te.compute(((n-bi)//si, (m-bj)//sj), lambda i, j: A[i*si+bi, j*sj+bj], name='b')
s = te.create_schedule(B.op)
mod = tvm.build(s, [A, B, bi, si, bj, sj])
```

Now test two cases to verify the correctness.

```{.python .input}
b = tvm.nd.array(np.empty((1, 3), dtype='float32'))
mod(a, b, 1, 2, 1, 1)
np.testing.assert_equal(b.asnumpy(), a.asnumpy()[1::2, 1::1])

b = tvm.nd.array(np.empty((1, 2), dtype='float32'))
mod(a, b, 2, 1, 0, 2)
np.testing.assert_equal(b.asnumpy(), a.asnumpy()[2::1, 0::2])
```

## Summary

- Both shape dimensions and indices can be expressions with variables.
- If a variable doesn't only appear in the shape tuple, we need to pass it as an argument when compiling.
