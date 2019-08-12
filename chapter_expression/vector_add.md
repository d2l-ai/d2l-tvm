# Add Two Vector 



```{.python .input  n=9}
import numpy as np
import tvm

def add(A, B, C):
    n = len(A)
    for i in range(n):
        C[i] = A[i] + B[i]
```

```{.python .input  n=58}
n = 100
a = np.random.normal(size=n).astype(np.float32)
b = np.random.normal(size=n).astype(np.float32)
c = np.zeros(shape=n, dtype=np.float32)
add(a, b, c)
```

## Implement in TVM

```{.python .input  n=60}
n = tvm.var("n")
A = tvm.placeholder((n,), name='A')
B = tvm.placeholder((n,), name='B')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
s = tvm.create_schedule(C.op)
fadd = tvm.build(s, [A, B, C])
```

```{.python .input  n=65}
type(fadd)
```

```{.python .input  n=61}
tvm.nd.array(a).dtype
```

```{.python .input  n=62}
e = tvm.nd.array(e)
fadd(tvm.nd.array(a), tvm.nd.array(b), e)
```

```{.python .input  n=82}
def eval_mod(mod, *args):
    tvm_args = [tvm.nd.array(arr) for arr in args]
    mod(*tvm_args)
    return [arr.asnumpy() for arr in tvm_args]

_, _, e = eval_mod(fadd, a, b, c)
e == c
```

```{.python .input  n=81}
try: 
    eval_mod(fadd, np.ones((1, 2)), np.ones((1, 2)), np.zeros((1, 2)))
except tvm.TVMError as e:
    print(e)
```

## Objects in TVM

```{.python .input  n=85}
type(A), type(C)
```

```{.python .input  n=86}
type(C.op)
```

```{.python .input  n=87}
type(s)
```

```{.python .input  n=88}
type(fadd)
```
