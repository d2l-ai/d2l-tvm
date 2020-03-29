# Conditional Expression: `if-then-else`
:label:`ch_if_then_else`

The `if-then-else` statement is supported through `te.if_then_else`. In this section, 
we will introduce this expression using computing the lower triangle of an matrix as the example.

```{.python .input  n=1}
import tvm
from tvm import te
import numpy as np
import d2ltvm
```

In NumPy, we can easily use `np.tril` to obtain the lower triangle.

```{.python .input  n=2}
a = np.arange(12, dtype='float32').reshape((3, 4))
np.tril(a)
```

Now let's implement it in TVM with `if_then_else`. It accepts three arguments, the first one is the condition, if true returning the second argument, otherwise returning the third one.

```{.python .input  n=3}
n, m = te.var('n'), te.var('m')
A = te.placeholder((n, m))
B = te.compute(A.shape, lambda i, j: te.if_then_else(i >= j, A[i,j], 0.0))

```

Verify the results.

```{.python .input  n=4}
b = tvm.nd.array(np.empty_like(a))
s = te.create_schedule(B.op)
print(tvm.lower(s, [A, B], simple_mode=True))
mod = tvm.build(s, [A, B])
mod(tvm.nd.array(a), b)
b
```

## Summary

- We can use `tvm.if_then_else` for the if-then-else statement.
