# Conditional Expressions: `if-then-else`

Lower triangle of an array.

```{.python .input  n=1}
import tvm
import numpy as np
import d2ltvm as d2l
```

```{.python .input  n=4}
a = np.arange(12, dtype='float32').reshape((3, 4))
np.tril(a)
```

```{.python .input  n=19}
n, m = tvm.var('n'), tvm.var('m')
A = tvm.placeholder((n, m))
B = tvm.compute(A.shape, lambda i, j: tvm.if_then_else(i >= j, A[i,j], 0.0))
s = tvm.create_schedule(B.op)
mod = tvm.build(s, [A, B])
```

```{.python .input  n=20}
b = np.empty_like(a)
d2l.eval_mod(mod, a, b)
b
```
