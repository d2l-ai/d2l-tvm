# Matrix Product

```{.python .input}
%matplotlib inline
import tvm
import numpy as np
import d2ltvm as d2l
```

```{.python .input}
def benchmark_one(func, n, use_tvm):
    x = np.random.normal(size=(n, n)).astype(np.float32)
    y = np.random.normal(size=(n, n)).astype(np.float32)
    z = np.empty_like(x)
    if use_tvm:
        x, y, z = tvm.nd.array(x), tvm.nd.array(y), tvm.nd.array(z)
    res = %timeit -o -q -r3 func(x, y, z)
    return 2 * n ** 3 / res.average / 1e9

def benchmark(func, use_tvm=True):
    gflops, sizes = [], 2**np.arange(5, 12, 1)#[512, 1024] #2**np.arange(5, 12, 1)
    np.random.seed(0)
    for n in sizes:
        gflops.append(benchmark_one(func, n, use_tvm))
    return sizes, gflops
```

```{.python .input}
def np_dot(x, y, z):
    np.dot(x, y, out=z)
    
sizes, np_gflops = benchmark(np_dot, use_tvm=False)
```

## Default Scheduling

```{.python .input}
# Save to the d2ltvm package.
def matrix_product(n):
    k = tvm.reduce_axis((0, n), name='k')
    A = tvm.placeholder((n, n), name='A')
    B = tvm.placeholder((n, n), name='B')
    C = tvm.compute(
        (n, n), lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k), name='C')
    s = tvm.create_schedule(C.op)
    return s, (A, B, C)
```

```{.python .input}
s, (A, B, C) = matrix_product(tvm.var('n'))

mod = tvm.build(s, [A, B, C])
print(tvm.lower(s, [A, B, C], simple_mode=True))
_, simple_gflops = benchmark(mod)
```

```{.python .input}
d2l.plot(sizes, [np_gflops, simple_gflops], xlabel='Matrix width/height', 
         xscale='log', ylabel='GFLOPS', yscale='log', 
         legend = ['numpy', 'default'])
```

## Reordering Axes

```{.python .input}
s = tvm.create_schedule(C.op)
(x, y), (k,) = C.op.axis, C.op.reduce_axis

s[C].reorder(x, k, y)
print(tvm.lower(s, [A, B, C], simple_mode=True))

mod = tvm.build(s, [A, B, C])
_, reorder_gflops = benchmark(mod)
```

```{.python .input}
d2l.plot(sizes, [np_gflops, simple_gflops, reorder_gflops], 
         xlabel='Matrix width/height', xscale='log', ylabel='GFLOPS', 
         yscale='log', legend = ['numpy', 'default', 'reorder'],
        fmts = ['--', '--', '-'])
```

## Parallelization

```{.python .input}
s = tvm.create_schedule(C.op)
(x, y), (k,) = C.op.axis, C.op.reduce_axis

s[C].reorder(x, k, y)
s[C].parallel(x)
print(tvm.lower(s, [A, B, C], simple_mode=True))

mod = tvm.build(s, [A, B, C])
_, parallel_gflops = benchmark(mod)
```

```{.python .input}
d2l.plot(sizes, [np_gflops, simple_gflops, reorder_gflops, parallel_gflops], 
         xlabel='Matrix width/height', xscale='log', ylabel='GFLOPS', 
         yscale='log', legend = ['numpy', 'default', 'reorder', '+ parallel'],
                fmts = ['--', '--', '--', '-'])
```

```{.python .input}

```

## Block Tiling

```{.python .input}
s = tvm.create_schedule(C.op)
(x, y), (k,) = C.op.axis, C.op.reduce_axis

cx = 2
cy = 16
xo, xi = s[C].split(x, cx)
yo, yi = s[C].split(y, cy)
s[C].reorder(xo, yo, k, xi, yi)
s[C].vectorize(yi)
s[C].unroll(xi)
s[C].parallel(xo)

mod = tvm.build(s, [A, B, C])
_, block_gflops = benchmark(mod)
```

```{.python .input}
d2l.plot(sizes, [np_gflops, simple_gflops, reorder_gflops, parallel_gflops, block_gflops], 
         xlabel='Matrix width/height', xscale='log', ylabel='GFLOPS', 
         yscale='log', legend = ['numpy', 'default', 'reorder', '+ parallel', '+ block'],
                fmts = ['--', '--', '--', '--', '-'])
```
