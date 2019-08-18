# Vector Addition
:label:`ch_vector_add_cpu`

In this section, we will optimize the vector addition implemented in :numref:`ch_vector_add` on CPU.

```{.python .input}
%matplotlib inline
import tvm
import numpy as np
from matplotlib import pyplot as plt
from IPython import display
```

First, we define a function to benchmark this operator using various length vectors. The magic build-in function `%timeit` is used to evaluate the running time. In which, we reduce the default 7 repeats into 3 to make the evaluation faster. The function returns the vector lengths and measured [GFLOPS](https://en.wikipedia.org/wiki/FLOPS), giga-floating point operations per second.

```{.python .input}
def benchmark(func, use_tvm=True):
    avg_times, sizes = [], (2**np.arange(5, 29, 4))
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

We also define a reusable plot function to draw multiple lines.

```{.python .input}
# Save to the d2ltvm package.
def plot(X, Y, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear', fmts=None,
         figsize=(6, 4)):
    """Plot multiple lines"""
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize
    axes = plt.gca()
    X, Y = np.array(X), np.array(Y)
    if X.shape != Y.shape: X = [X] * len(Y)
    if not fmts: fmts = ['-'] * len(X)
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend: axes.legend(legend)
    axes.grid()
```

We will use NumPy as the performance baseline. The output ndarray is passed through `out` to avoid unnecessary memory copy.

```{.python .input}
def np_add(x, y, z):
    np.add(x, y, out=z)
    
sizes, np_gflops = benchmark(np_add, use_tvm=False)

plot(sizes, [np_gflops], xlabel='Vector length', xscale='log', 
     ylabel='GFLOPS', yscale='log', legend = ['numpy'])
```

As we can see that the performance first increases with the vector length, which due to the system overhead dominates when the workload is small. The performance then decreases when we cannot fit all data into the L3 cache.

## Default Scheduling

Let's first copy the vector addition expression from :numref:`ch_vector_add`

```{.python .input}
n = tvm.var('n')
A = tvm.placeholder((n,), name='a')
B = tvm.placeholder((n,), name='b')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='c')
```

Then benchmark the default scheduling, which generate a single for-loop program. In default, TVM will generate LLVM codes and compile to machine code by LLVM.

```{.python .input}
s = tvm.create_schedule(C.op)
print(tvm.lower(s, [A, B, C], simple_mode=True))

mod = tvm.build(s, [A, B, C])
_, simple_gflops = benchmark(mod)
```

From the following figure, we can see that the default scheduling is as good as NumPy, and even outperforms when the workloads are small.

```{.python .input}
plot(sizes, [np_gflops, simple_gflops], xlabel='Vector length', xscale='log', 
     ylabel='GFLOPS', yscale='log', legend = ['numpy', 'default'], fmts=['--', '-'])
```

## Parallelization

For such a simple program, e.g. sequentially read/write with a single for-loop, LLVM is able to generate highly efficient machine codes. One important optimization that is not enabled in default is parallelization. The vector addition operator is [embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel), we can just change the for-loop into a parallel for-loop. In TVM, we first obtain the scheduler for `C` by `s[C]`, and then require to parallel the computation of the first axis, which is `C.op.axis[0]`.

```{.python .input}
s = tvm.create_schedule(C.op)
s[C].parallel(C.op.axis[0])
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

We can see that `for` is changed to `parallel` in the pseudo codes. Comparing the results we obtained before, parallelization significantly improve the performance when the workloads are large, e.g. vector lengths beyond $10^4$. Though the parallelization overhead impact the performance for small workloads, where single thread is even faster.

```{.python .input}
mod = tvm.build(s, [A, B, C])
_, parallel_gflops = benchmark(mod)

plot(sizes, [np_gflops, simple_gflops, parallel_gflops],
     xlabel='Vector length', xscale='log', ylabel='GFLOPS', yscale='log', 
     legend = ['numpy', 'default', 'parallel'], fmts=['--', '--', '-'])
```

## Summary

- The default scheduling generates highly efficient single-thread CPU program.
- Parallelization improves performance for large workloads.
