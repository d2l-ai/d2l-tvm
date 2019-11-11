# Truth Value Testing: `all` and `any`

In Python, we can use `all` and `any` to get the boolean return of a list of values. `all` returns the logical `and` result while `any` returns the logical `or` result.

```{.python .input}
import numpy as np
import d2ltvm 
import tvm

any((0, 1, 2)), all((0, 1, 2))
```

TVM provides similar `tvm.all` and `tvm.any`, which are useful to construct complex conditional expression for `tvm.if_then_else`. 

The example we will use is padding the matrix `a` with 0s.

```{.python .input  n=3}
a = np.ones((3, 4), dtype='float32')
# applying a zero padding of size 1 to a
b = np.zeros((5, 6), dtype='float32')
b[1:-1,1:-1] = a
print(b)
```

Now let's implement it in TVM. Note that we pass the four condition values into `tvm.any`. 

```{.python .input}
p = 1 # padding size
n, m = tvm.var('n'), tvm.var('m')
A = tvm.placeholder((n, m), name='a')
B = tvm.compute((n+p*2, m+p*2),
                lambda i, j: tvm.if_then_else(
                    tvm.any(i<p, i>=n+p, j<p, j>=m+p), 0, A[i-p, j-p]),
                name='b')
```

Verify the results.

```{.python .input}
s = tvm.create_schedule(B.op)
mod = tvm.build(s, [A, B])
c = tvm.nd.array(np.empty_like(b))
mod(tvm.nd.array(a), c)
print(c)
```

## Summary

- We can use `tvm.any` and `tvm.all` to construct complex conditional expressions.
