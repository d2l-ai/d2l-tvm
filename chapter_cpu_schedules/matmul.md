# Matrix Multiplication
:label:`ch_matmul_cpu`

We saw the NumPy `dot` operator nearly reaches the peak performance of the Xeon E5-2686 v4 CPU in :numref:`ch_cpu_arch`. In this chapter, we will investigate multiple scheduling strategies for this operator.

## Setup

```{.python .input  n=1}
%matplotlib inline
import d2ltvm
import numpy as np
import timeit
import tvm

target = 'llvm -mcpu=core-avx2'
```

As we did in :numref:`ch_vector_add_cpu`, we first define a function to measure NumPy performance as our baseline.

```{.python .input  n=2}
# Save to the d2ltvm package.
def np_matmul_timer(n):
    timer = timeit.Timer(setup='import numpy as np\n'
                         'import d2ltvm\n'
                         'a, b, c = d2ltvm.get_abc(%s)' % str((n,n)),
                         stmt = 'np.dot(a, b, out=c)')
    return timer.timeit

sizes = 2**np.arange(5, 12, 1)
np_times = [d2ltvm.bench_workload(np_matmul_timer(n)) for n in sizes]
np_gflops = 2 * sizes **3 / 1e9 / np.array(np_times)
```

## The Default Schedule

The default schedule consists of three for loops.

```{.python .input  n=3}
def default(n):
    A, B, C = d2ltvm.matmul(n, n, n)
    return tvm.create_schedule(C.op), (A, B, C)

s, args = default(64)
print(tvm.lower(s, args, simple_mode=True))
```

To benchmark its performance, we also define a reusable function as we did in :numref:`ch_vector_add_cpu`.

```{.python .input  n=4}
# Save to the d2ltvm package.
def bench_matmul_tvm(func, sizes, target):
    def workload(nrepeats):
        timer = mod.time_evaluator(mod.entry_name, ctx=ctx, number=nrepeats)
        return timer(a, b, c).mean * nrepeats
    times = []
    for n in sizes:
        s, (A, B, C) = func(int(n))
        mod = tvm.build(s, [A, B, C], target)
        ctx = tvm.context(target, 0)
        a, b, c = d2ltvm.get_abc((n, n), lambda x: tvm.nd.array(x, ctx=ctx))
        times.append(d2ltvm.bench_workload(workload))
    return 2 * sizes**3 / 1e9 / np.array(times)
```

The default schedule follows the computation illustrated in :numref:`fig_matmul_default`.
It's not surprised to see that the default schedule doesn't perform well, especially on large matrices that cannot fit into the cache.

```{.python .input  n=5}
default_gflops = bench_matmul_tvm(default, sizes, target)
d2ltvm.plot_gflops(sizes, [np_gflops, default_gflops], ['numpy', 'default'])
```

## Reordering Axes

The first problem we can see from :numref:`fig_matmul_default` is that $B$ is accessed by columns while its elements are stored by rows. The reason is because we iterate axis `y` before axis `k`. Simply switching these two for-loops will make all elements read and write sequential. :numref:`fig_matmul_reorder` illustrates the changed the data access pattern.

![Reorder axes in matrix multiplication.](../img/matmul_reorder.svg)
:label:`fig_matmul_reorder`

To implement it, we change the axes order from (`x`, `y`, `k`) to (`x`, `k`, `y`) by the `reorder` method.

```{.python .input  n=6}
def reorder(n):
    s, (A, B, C) = default(n)
    (x, y), (k,) = C.op.axis, C.op.reduce_axis
    s[C].reorder(x, k, y)
    return s, (A, B, C)

s, args = reorder(64)
print(tvm.lower(s, args, simple_mode=True))
```

We can see that the reordering significantly improves the performance compared to the default schedule.

```{.python .input  n=7}
reorder_gflops = bench_matmul_tvm(reorder, sizes, target)
d2ltvm.plot_gflops(sizes, [np_gflops, default_gflops, reorder_gflops],
            ['numpy', 'default', 'reorder'])
```

## Parallelization

In the outermost for-loop, each time we compute the results of a row in $C$. Each row can be computed in parallel, so we can make the schedule be parallelized on axis `x`.

```{.python .input  n=8}
def parallel(n):
    s, (A, B, C) = reorder(n)
    s[C].parallel(C.op.axis[0])
    return s, (A, B, C)

s, args = parallel(64)
print(tvm.lower(s, args, simple_mode=True))
```

```{.python .input  n=9}
parallel_gflops = bench_matmul_tvm(parallel, sizes, target)
d2ltvm.plot_gflops(sizes, [np_gflops, default_gflops, reorder_gflops, parallel_gflops],
            ['numpy', 'default', 'reorder', 'parallel'])
```

Parallelization improves the performance again. But we can see that there is still a gap compared to NumPy on large matrices, specially when they cannot fit into the L2 cache. We will try to resolve it in the next chapter.

## Summary

1. Reordering the for-loops in matrix multiplication properly improves the performance.
1. Parallelization is also important.

## Exercises

1. Change the number of threads
1. Try a different axes order in function `parallel`
1. Benchmark larger matrix sizes
