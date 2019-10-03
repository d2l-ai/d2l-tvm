# Matrix Multiplication
:label:`ch_matmul_cpu`

We mentioned in :numref:`ch_cpu_arch` that matrix multiplication is a widely used performance benchmark workload, and the NumPy `dot` operator nearly reaches the peak performance of the Xeon E5-2686 v4 CPU. In this chapter, we will investigate multiple scheduling strategies for this operator.

```{.python .input  n=1}
%matplotlib inline
import tvm
import numpy as np
import d2ltvm as d2l
```

We first define benchmark functions to measure the GFLOPS. To simplify the measurement, we only consider square matrices. Extending to non-square cases is straightforward. Then let's reproduce the matrix multiplication result in :numref:`ch_cpu_arch`.

```{.python .input  n=2}
def benchmark_square_matmul(func, n, constructor=None):
    x = np.random.normal(size=(n, n)).astype(np.float32)
    y = np.random.normal(size=(n, n)).astype(np.float32)
    z = np.empty_like(x)
    if constructor:
        x, y, z = constructor(x), constructor(y), constructor(z)
    res = %timeit -o -q -r3 func(x, y, z)
    return 2 * n ** 3 / res.average / 1e9

def np_dot(x, y, z):
    np.dot(x, y, out=z)
    
benchmark_square_matmul(np_dot, 1024)
```

Now we benchmark the performance on various input sizes as our baseline. Note that the `func` accepts the matrix size to return a benchmark function.

```{.python .input  n=3}
def benchmark(func, constructor=None):
    gflops, sizes = [], (2**np.arange(5, 14, 1)).tolist()
    np.random.seed(0)
    for n in sizes:
        gflops.append(benchmark_square_matmul(func(n), n, constructor))
    return sizes, gflops

sizes, np_gflops = benchmark(lambda n: np_dot)
```

## Default Schedule

Given $A, B \in\mathbb R^{n\times n}$, if $C=AB$ then 

$$C_{x,y} = \sum_{k=1}^n A_{x,k} B_{k,y}.$$

The elements assessed to compute $C_{i,j}$ are illustrated in :numref:`fig_matmul_default`. 

![Compute $C_{x,y}$ in matrix multiplication.](../img/matmul_default.svg)
:label:`fig_matmul_default`

The following function returns the computing expression of matrix multiplication.

```{.python .input  n=4}
# Save to the d2ltvm package.
def square_matmul(n):
    """Return the computing expression of square matrix multiplication. 
    """
    k = tvm.reduce_axis((0, n), name='k')
    A = tvm.placeholder((n, n), name='A')
    B = tvm.placeholder((n, n), name='B')
    C = tvm.compute(
        (n, n), lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k), name='C')
    return (A, B, C)
```

An operator has a better performance when the shape is known before compilation. That's passing `n=1024` performs better than `n=tvm.var()`. Therefore, we will compile a module for each shape. The following function returns a function that can be used by `benchmark`.

```{.python .input  n=22}
# Save to the d2ltvm package.
def square_matmul_module(schedule_updater=None, 
                         target='llvm -mcpu=core-avx2'):
    """Returns a function that accepts the input size n to return
    a TVM module. 
    """
    def func(n):
        A, B, C = square_matmul(n)
        s = tvm.create_schedule(C.op)
        if schedule_updater is not None: 
            schedule_updater(s, C)
        mod = tvm.build(s, [A, B, C], target=target)
        return mod
    return func

_, default_gflops = benchmark(square_matmul_module(), tvm.nd.array)
```

The default schedule follows the computation illustrated in :numref:`fig_matmul_default`.
It's not surprised to see that the default schedule doesn't perform well, especially on large matrices that cannot fit into the cache.

```{.python .input  n=7}
def plot(gflops, legend):
    d2l.plot(sizes, gflops, xlabel='Matrix width/height', ylabel='GFLOPS', 
             xscale='log', yscale='log', 
             legend=legend, fmts=['--']*(len(gflops)-1)+['-'])
    
plot([np_gflops, default_gflops], ['numpy', 'default'])
```

## Reordering Axes

The first problem we can see from :numref:`fig_matmul_default` is that $B$ is accessed by columns while its elements are stored by rows. The reason is because we iterate axis `y` before axis `k`. Simply switching these two for-loops will make all elements read and write sequential. :numref:`fig_matmul_reorder` illustrates the changed the data access pattern. 

![Reorder axes in matrix multiplication.](../img/matmul_reorder.svg)
:label:`fig_matmul_reorder`

To implement it, we change the axes order from (`x`, `y`, `k`) to (`x`, `k`, `y`) by the `reorder` method.

```{.python .input  n=8}
def reorder(s, C):
    (x, y), (k,) = C.op.axis, C.op.reduce_axis
    s[C].reorder(x, k, y)
    
_, reorder_gflops = benchmark(square_matmul_module(reorder), tvm.nd.array) 

plot([np_gflops, default_gflops, reorder_gflops], 
     ['numpy', 'default', 'reorder'])
```

We can see that the reordering significantly improves the performance compared to the default schedule.

## Parallelization

In the outermost for-loop, each time we compute the results of a row in $C$. Each row can be computed in parallel, so we can make the schedule be parallelized on axis `x`. As discussed in :numref:`ch_cpu_arch`, despite our OS claims there are 32 threads, our CPU only has 16 cores.

```{.python .input  n=10}
import os 
os.environ["TVM_NUM_THREADS"] = '16'

def parallel(s, C):
    (x, y), (k,) = C.op.axis, C.op.reduce_axis
    s[C].reorder(x, k, y)
    s[C].parallel(x)
    
_, parallel_gflops = benchmark(square_matmul_module(parallel), tvm.nd.array)    

plot([np_gflops, default_gflops, reorder_gflops, parallel_gflops], 
     ['numpy', 'default', 'reorder', '+parallel'])
```

Parallelization improves the performance again. But we can see that there is still a gap compared to NumPy on large matrices, specially when they cannot fit into the L2 cache. We will try to resolve it in the next chapter.  

## Summary

1. Reordering the for-loops in matrix multiplication properly improves the performance. 
1. Parallelization is also important.

## Exercises

1. Change the number of threads
1. Try a different axes order in function `parallel` 
1. Benchmark larger matrix sizes
