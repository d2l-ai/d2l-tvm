# Conditional Expression: `if-then-else`
:label:`ch_if_then_else`

The `if-then-else` statement is supported trough `tvm.if_then_else`. In this chapter, we will use computing the lower triangle of an matrix as the example to introduce this expression.

```{.python .input  n=1}
import tvm
import numpy as np
import d2ltvm
```

In NumPy, we can use `np.tril` to obtain the lower triangle.

```{.python .input  n=2}
a = np.arange(12, dtype='float32').reshape((3, 4))
np.tril(a)
```

Now let's implement it in TVM with `if_then_else`. It accepts three arguments, the first one is the condition, if true then returns the second argument, otherwise returns the third one.

```{.python .input  n=3}
n, m = tvm.var('n'), tvm.var('m')
A = tvm.placeholder((n, m))
B = tvm.compute(A.shape, lambda i, j: tvm.if_then_else(i >= j, A[i,j], 0.0))

```

Verify the results.

```{.python .input  n=4}
b = tvm.nd.array(np.empty_like(a))
s = tvm.create_schedule(B.op)
print(tvm.lower(s, [A, B], simple_mode=True))
mod = tvm.build(s, [A, B])
mod(tvm.nd.array(a), b)
b
```

## Summary

- We can use `tvm.if_then_else`  for the if-then-else statement.
