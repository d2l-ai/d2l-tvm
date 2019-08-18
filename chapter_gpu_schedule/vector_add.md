# Vector Addition

```{.python .input}
%matplotlib inline
import tvm
import numpy as np
import d2ltvm as d2l
import mxnet as mx
```

```{.python .input}
def benchmark(func, use_tvm=True):
    avg_times, sizes = [], 2**np.arange(10, 30, 3)
    np.random.seed(0)
    for size in sizes:
        x = np.random.normal(size=size).astype(np.float32)
        y = np.random.normal(size=size).astype(np.float32)
        z = np.empty_like(x)
        if use_tvm:
            x, y, z = [tvm.nd.array(a, ctx=tvm.gpu(0)) for a in [x, y, z]] 
        else:
            x, y, z = [mx.nd.array(a, ctx=mx.gpu(0)) for a in [x, y, z]]
        res = %timeit -o -q -r3 func(x, y, z)
        avg_times.append(res.average)
    return sizes, sizes * 2 / avg_times / 1e9
```

```{.python .input}
def mx_add(x, y, z):
    mx.nd.elemwise_add(x, y, out=z)
    z.wait_to_read()
    
sizes, mx_gflops = benchmark(mx_add, use_tvm=False)
```

```{.python .input}
n = tvm.var('n')
A = tvm.placeholder((n,), name='a')
B = tvm.placeholder((n,), name='b')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='c')
```

```{.python .input}
s = tvm.create_schedule(C.op)
bx, tx = s[C].split(C.op.axis[0], factor=128)
s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
mod = tvm.build(s, [A, B, C], 'cuda')
```

```{.python .input}
_, tvm_gflops = benchmark(mod)
d2l.plot(sizes, [mx_gflops, tvm_gflops], xlabel='Vector length', xscale='log', 
     ylabel='GFLOPS', yscale='log', legend = ['mxnet', 'tvm'], fmts=['-', '-'])
```
