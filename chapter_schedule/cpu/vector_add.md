# Optimize Vector Addition

```{.python .input}
%matplotlib inline
import tvm
import numpy as np
from matplotlib import pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')
```

```{.python .input}
def benchmark(func, use_tvm=True):
    avg_times, sizes = [], (2**np.arange(5, 29, 2))
    np.random.seed(0)
    for size in sizes:
        x = np.random.normal(size=size).astype(np.float32)
        y = np.random.normal(size=size).astype(np.float32)
        z = np.empty_like(x)
        if use_tvm:
            x, y, z = tvm.nd.array(x), tvm.nd.array(y), tvm.nd.array(z)
        res = %timeit -o -q -r3 func(x, y, z)
        avg_times.append(res.average)
    return sizes, sizes * 2 / avg_times / 1e9
```

```{.python .input}
def np_add(x, y, z):
    z[:] = x + y
sizes, np_gflops = benchmark(np_add, use_tvm=False)
```

## Default Scheduling

```{.python .input}
n = tvm.var()
A = tvm.placeholder((n,))
B = tvm.placeholder((n,))
C = tvm.compute(A.shape, lambda i: A[i] + B[i])
```

```{.python .input}
s = tvm.create_schedule(C.op)
mod = tvm.build(s, [A, B, C])
_, simple_gflops = benchmark(mod)
```

## Parallelization

```{.python .input}
s = tvm.create_schedule(C.op)
s[C].parallel(C.op.axis[0])
mod = tvm.build(s, [A, B, C])
_, parallel_gflops = benchmark(mod)
```

## Results Comparison

```{.python .input}
def plot(X, Y, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         figsize=(6, 4)):
    """Plot multiple lines"""
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize
    axes = plt.gca()
    if len(X) != len(Y): X = [X] * len(Y)
    for x, y in zip(X, Y):
        axes.plot(x, y)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend: axes.legend(legend)
    axes.grid()
```

```{.python .input}
plot(sizes, [np_gflops, simple_gflops, parallel_gflops], 
     xlabel='vector length', xscale='log',
     ylabel='gflops', yscale='log', 
     legend = ['numpy', 'naive', 'parallel'])
```

## Summary

- parallel 
