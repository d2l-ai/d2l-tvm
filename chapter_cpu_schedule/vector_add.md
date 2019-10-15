# Vector Addition
:label:`ch_vector_add_cpu`

In this section, we will optimize the vector addition implemented in :numref:`ch_vector_add` on CPU.

```{.python .input  n=31}
%matplotlib inline
import d2ltvm
import inspect
from IPython import display
import numpy as np
from matplotlib import pyplot as plt
import timeit
import tvm
```

We first define reusable plot functions to draw multiple lines, which generalize the plot function defined in :numref:`ch_call_overhead`.

```{.python .input  n=28}
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
    
# Save to the d2ltvm package
def plot_gflops(sizes, gflops, legend):
    d2ltvm.plot(sizes, gflops, xlabel='Size', ylabel='GFLOPS', 
             xscale='log', yscale='log', 
             legend=legend, fmts=['--']*(len(gflops)-1)+['-'])
```

Then we define functions to return the setup string that create input and output ndarrays.

```{.python .input  n=34}
# Save to the d2ltvm package.
def get_xyz(shape):
    np.random.seed(0)
    x = np.random.normal(size=shape).astype(np.float32)
    y = np.random.normal(size=shape).astype(np.float32)
    z = np.empty_like(x)
    return x, y, z

# Save to the d2ltvm package.
def np_setup_xyz(shape):
    return 'import numpy as np\n' + \
           inspect.getsource(get_xyz) + \
           'x, y, z = get_xyz(%s)' % str(shape)
```

to benchmark this operator using various length vectors. The magic build-in function `%timeit` is used to evaluate the running time. In which, we reduce the default 7 repeats into 3 to make the evaluation faster. The function returns the vector lengths and measured [GFLOPS](https://en.wikipedia.org/wiki/FLOPS), giga-floating point operations per second.

```{.python .input  n=35}
sizes = 2**np.arange(5, 29, 4)
np_add = lambda n: timeit.Timer(setup=np_setup_xyz(n),
                                stmt='np.add(x, y, out=z)')    
np_times = [d2ltvm.bench_workload(np_add(n).timeit) for n in sizes]
np_gflops = 2 * sizes / 1e9 /np.array(np_times)
plot_gflops(sizes, [np_gflops], ['numpy'])
```

As we can see that the performance first increases with the vector length, which due to the system overhead dominates when the workload is small. The performance then decreases when we cannot fit all data into the L3 cache.

## Default Scheduling

Let's first copy the vector addition expression from :numref:`ch_vector_add`

```{.python .input  n=5}
n = tvm.var('n')
A = tvm.placeholder((n,), name='a')
B = tvm.placeholder((n,), name='b')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='c')
```

Then benchmark the default scheduling, which generate a single for-loop program. In default, TVM will generate LLVM codes and compile to machine code by LLVM.

```{.python .input  n=6}
s = tvm.create_schedule(C.op)
target = 'llvm -mcpu=core-avx2'
print(tvm.lower(s, [A, B, C], simple_mode=True))

def bench_tvm(s):
    mod = tvm.build(s, [A, B, C], target)
    ctx = tvm.context(target, 0)
    def workload(nrepeats):
        timer = mod.time_evaluator(mod.entry_name, ctx=ctx, number=nrepeats)
        return timer(x, y, z).mean * nrepeats
    times = []
    for n in sizes:
        x, y, z = [tvm.nd.array(a) for a in get_xyz(n)]
        times.append(d2ltvm.bench_workload(workload))
    return 2 * sizes / 1e9 /np.array(times)

simple_gflops = bench_tvm(s)
```

From the following figure, we can see that the default scheduling is as good as NumPy, and even outperforms when the workloads are small.

```{.python .input  n=7}
plot_gflops(sizes, [np_gflops, simple_gflops], ['numpy', 'default'])
```

## Parallelization

For such a simple program, e.g. sequentially read/write with a single for-loop, LLVM is able to generate highly efficient machine codes. One important optimization that is not enabled in default is parallelization. The vector addition operator is [embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel), we can just change the for-loop into a parallel for-loop. In TVM, we first obtain the scheduler for `C` by `s[C]`, and then require to parallel the computation of the first axis, which is `C.op.axis[0]`.

```{.python .input  n=8}
s = tvm.create_schedule(C.op)
s[C].parallel(C.op.axis[0])
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

We can see that `for` is changed to `parallel` in the pseudo codes. Comparing the results we obtained before, parallelization significantly improve the performance when the workloads are large, e.g. vector lengths beyond $10^4$. Though the parallelization overhead impact the performance for small workloads, where single thread is even faster.

```{.python .input  n=9}
parallel_gflops = bench_tvm(s)

plot_gflops(sizes, [np_gflops, simple_gflops, parallel_gflops],
     ['numpy', 'default', 'parallel'])
```

## Summary

- The default scheduling generates highly efficient single-thread CPU program.
- Parallelization improves performance for large workloads.
