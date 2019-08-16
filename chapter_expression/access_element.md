# Accessing Elements

We have already seen how to access element sequentially by `a[i]` or `a[i,j]`, and slice the whole axis by with a reduction axis. In this section, we will show more flexible ways to access elements. 

## Matrix Transpose

Our first example is matrix transpose `a.T`, in which we access `a`'s elements by columns.

```{.python .input  n=1}
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

```{.json .output n=1}
[
 {
  "data": {
   "text/plain": "produce b {\n  for (i, 0, m) {\n    for (j, 0, n) {\n      b[((i*n) + j)] = a[((j*m) + i)]\n    }\n  }\n}"
  },
  "execution_count": 1,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

The computation equals to `b[i,j] = a[j,i]`, but note how TVM expands the 2-D indices to 1-D. 

The results should be as expected.

```{.python .input  n=2}
a = np.random.normal(size=(3, 4)).astype('float32')
b = np.empty((4, 3), dtype='float32')

mod = tvm.build(s, [A, B])
d2l.eval_mod(mod, a, b)
np.testing.assert_equal(b, a.T)
```

## Reshaping 

Next let's use expressions for indexing. The following codes reshape a 2-D array `a` to 1-D by `a.reshape(-1)`. Note that how we convert the 1-D index `i` to the 2-D index `[i/m, i%m]`.

```{.python .input  n=3}
B = tvm.compute((m*n, ), lambda i: A[i/m, i%m], name='B')
s = tvm.create_schedule(B.op)
tvm.lower(s, [A, B], simple_mode=True)
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "produce B {\n  for (i, 0, (m*n)) {\n    B[i] = a[i]\n  }\n}"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Since internally a $n$-D array is treated as a 1-D array, the generated codes simplify the index expression `(i/m)*m + i%m` to `i` to improve the efficiency. 

We can implement a general 2-D reshape function as well.

```{.python .input  n=8}
p, q = tvm.var('p'), tvm.var('q')
B = tvm.compute((p, q), lambda i, j: A[(i*q+j)/m, (i*q+j)%m], name='B')
s = tvm.create_schedule(B.op)
tvm.lower(s, [A, B], simple_mode=True)
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "produce B {\n  for (i, 0, p) {\n    for (j, 0, q) {\n      B[((i*q) + j)] = a[(j + (i*q))]\n    }\n  }\n}"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Also note that index expressions are simplified when converting from 2-D to 1-D. 

When testing the results, we should be aware that we put no constraint on the output shape, which can have an arbitrary shape `(p, q)`, and therefore TVM will not be able to check $qp = nm$ for us. For example, in the following example we created a `b` with size (20) larger than `a` (12), then only the first 12 elements in `b` are from `a`. 

```{.python .input  n=19}
mod = tvm.build(s, [A, B])
a = np.arange(12, dtype='float32').reshape((3,4))
b = np.zeros((5, 4), dtype='float32')
d2l.eval_mod(mod, a, b)
b
```

```{.json .output n=19}
[
 {
  "data": {
   "text/plain": "array([[ 0.0000000e+00,  1.0000000e+00,  2.0000000e+00,  3.0000000e+00],\n       [ 4.0000000e+00,  5.0000000e+00,  6.0000000e+00,  7.0000000e+00],\n       [ 8.0000000e+00,  9.0000000e+00,  1.0000000e+01,  1.1000000e+01],\n       [ 1.3452465e-43,  0.0000000e+00,  1.1350518e-43,  0.0000000e+00],\n       [-1.2934314e+34,  3.0887421e-41, -1.0389684e+34,  3.0887421e-41]],\n      dtype=float32)"
  },
  "execution_count": 19,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Summary


