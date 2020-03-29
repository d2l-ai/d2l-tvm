# Truth Value Testing: `all` and `any`
:label:`ch_all_any`

In Python, we can use `all` and `any` to get the boolean return of a list of values. `all` returns the logical `and` result while `any` returns the logical `or` result.

```{.python .input}
import numpy as np
import d2ltvm 
import tvm
from tvm import te

any((0, 1, 2)), all((0, 1, 2))
```

TVM provides similar `te.all` and `te.any`, which are useful to construct complex conditional expression for `te.if_then_else`. 

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
n, m = te.var('n'), te.var('m')
A = te.placeholder((n, m), name='a')
B = te.compute((n+p*2, m+p*2),
                lambda i, j: te.if_then_else(
                    te.any(i<p, i>=n+p, j<p, j>=m+p), 0, A[i-p, j-p]),
                name='b')
```

Verify the results.

```{.python .input}
s = te.create_schedule(B.op)
mod = tvm.build(s, [A, B])
c = tvm.nd.array(np.empty_like(b))
mod(tvm.nd.array(a), c)
print(c)
```

## Summary

- We can use `tvm.any` and `tvm.all` to construct complex conditional expressions.
