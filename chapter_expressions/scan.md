# Looping: `scan`

Though we cannot use `for`-loop directly in TVM, the alternative solution is through `tvm.scan`.

```{.python .input  n=1}
import d2ltvm
import numpy as np
import tvm
```

Fibonacci number

```{.python .input  n=5}
n = tvm.var('n')
state = tvm.placeholder((n, ), name='x')
init = tvm.compute((2,), lambda _: 1.0)
update = tvm.compute((n,), lambda i: state[i-1]+state[i-2])
A = tvm.scan(init, update, state)
s = tvm.create_schedule(A.op)
mod = tvm.build(s, [A])
tvm.lower(s, [A], simple_mode=True)
```

```{.python .input}
a = np.empty((10,), dtype='float32')
d2l.eval_mod(mod, out=a)
a
```

Cumulative sum
