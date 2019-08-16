# Looping: `scan`

```{.python .input}
import tvm
import numpy as np
import d2ltvm as d2l
```

Fibonacci number


```{.python .input}
n = tvm.var('n')
state = tvm.placeholder((n, ))
init = tvm.compute((2,), lambda _: 1.0)
update = tvm.compute((n,), lambda i: state[i-1]+state[i-2])
A = tvm.scan(init, update, state)
s = tvm.create_schedule(A.op)
mod = tvm.build(s, [A])
```

```{.python .input}
a = np.empty((10,), dtype='float32')
d2l.eval_mod(mod, a)
a
```

Cumulative sum

