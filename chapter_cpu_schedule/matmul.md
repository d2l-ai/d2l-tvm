# Matrix Multiplication
:label:`ch_matmul_cpu`

We mentioned in :numref:`ch_cpu_arch` that matrix multiplication is a widely used performance benchmark workload, and the NumPy `dot` operator nearly reaches the peak performance of the Xeon E5-2686 v4 CPU. In this chapter, we will investigate multiple scheduling strategies for this operator.

```{.python .input  n=2}
%matplotlib inline
import tvm
import numpy as np
import d2ltvm 
import timeit
import time
```

We first define benchmark functions to measure the GFLOPS. To simplify the measurement, we only consider square matrices. Extending to non-square cases is straightforward. Then let's reproduce the matrix multiplication result in :numref:`ch_cpu_arch`.

```{.python .input  n=6}
# Save to the d2ltvm package.
def benchmark_square_matmul_np(n):
    timer = timeit.Timer(
        setup='import numpy as np\n'
        'n = ' + str(n) + '\n'
        'x = np.random.normal(size=(n, n)).astype(np.float32)\n'
        'y = np.random.normal(size=(n, n)).astype(np.float32)\n'
        'z = np.empty_like(x)\n',
        stmt = 'np.dot(x, y, out=z);')
    # Estimate the #repeat to run for 1 second
    time = timer.timeit(1)
    nrepeat = max(int(1.0/time), 5) 
    time = timer.timeit(nrepeat) 
    return 2 * n**3 / time / 1e9 * nrepeat

benchmark_square_matmul_np(1024)
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "517.7183194437663"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Next we define a function to benchmark multiple input shapes.

```{.python .input  n=21}
sizes = 2**np.arange(5, 12, 1)
np_gflops = [benchmark_square_matmul_np(n) for n in sizes]
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
def square_matmul_default(n):
    """Return the computing expression of square matrix multiplication with
    the default schedule.
    """
    k = tvm.reduce_axis((0, n), name='k')
    A = tvm.placeholder((n, n), name='A')
    B = tvm.placeholder((n, n), name='B')
    C = tvm.compute(
        (n, n), lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k), name='C')
    return tvm.create_schedule(C.op), (A, B, C)
```

Now let's check the performance of the default schedule. Note that an operator has a better performance when the shape is known before compilation. That's passing `n=1024` performs better than `n=tvm.var()`. Therefore, we will compile a module for each shape. 

The following function returns a function that can be used by `benchmark`.

```{.python .input  n=22}
# Save to the d2ltvm package.
def benchmark_square_matmul_tvm(n, generator, target='llvm -mcpu=core-avx2'):
    # Compile
    s, [A, B, C] = generator(int(n))
    mod = tvm.build(s, [A, B, C], target=target)
    # Prepare inputs and outputs
    x = np.random.normal(size=(n, n)).astype(np.float32)
    y = np.random.normal(size=(n, n)).astype(np.float32)
    z = np.empty_like(x)
    ctx = tvm.context(target, 0)
    x, y, z = tvm.nd.array(x, ctx), tvm.nd.array(y, ctx), tvm.nd.array(z, ctx)
    # Estimate the #runs to roughly benchmark for 1 second
    start = time.time()
    mod(x, y, z)
    nrepeat = int(max(1.0/(time.time() - start), 1))
    timer = mod.time_evaluator(mod.entry_name, ctx=ctx, number=nrepeat)
    return 2 * n**3 / timer(x, y, z).mean / 1e9

default_gflops = [benchmark_square_matmul_tvm(n, square_matmul_default) for n in sizes]
```

The default schedule follows the computation illustrated in :numref:`fig_matmul_default`.
It's not surprised to see that the default schedule doesn't perform well, especially on large matrices that cannot fit into the cache.

```{.python .input  n=7}
# Save to the d2ltvm package
def plot_gflops(sizes, gflops, legend):
    d2ltvm.plot(sizes, gflops, xlabel='Size', ylabel='GFLOPS', 
             xscale='log', yscale='log', 
             legend=legend, fmts=['--']*(len(gflops)-1)+['-'])
    
plot_gflops(sizes, [np_gflops, default_gflops], ['numpy', 'default'])
```

## Reordering Axes

The first problem we can see from :numref:`fig_matmul_default` is that $B$ is accessed by columns while its elements are stored by rows. The reason is because we iterate axis `y` before axis `k`. Simply switching these two for-loops will make all elements read and write sequential. :numref:`fig_matmul_reorder` illustrates the changed the data access pattern. 

![Reorder axes in matrix multiplication.](../img/matmul_reorder.svg)
:label:`fig_matmul_reorder`

To implement it, we change the axes order from (`x`, `y`, `k`) to (`x`, `k`, `y`) by the `reorder` method.

```{.python .input  n=8}
def reorder(n):
    s, (A, B, C) = square_matmul_default(n)
    (x, y), (k,) = C.op.axis, C.op.reduce_axis
    s[C].reorder(x, k, y)
    return s, (A, B, C)

reorder_gflops = [benchmark_square_matmul_tvm(n, reorder) for n in sizes]

plot_gflops(sizes, [np_gflops, default_gflops, reorder_gflops], 
            ['numpy', 'default', 'reorder'])
```

We can see that the reordering significantly improves the performance compared to the default schedule.

## Parallelization

In the outermost for-loop, each time we compute the results of a row in $C$. Each row can be computed in parallel, so we can make the schedule be parallelized on axis `x`. As discussed in :numref:`ch_cpu_arch`, despite our OS claims there are 32 threads, our CPU only has 16 cores.

```{.python .input  n=10}
import os 
os.environ["TVM_NUM_THREADS"] = '16'

def parallel(n):
    s, (A, B, C) = square_matmul_default(n)
    (x, y), (k,) = C.op.axis, C.op.reduce_axis
    s[C].reorder(x, k, y)
    s[C].parallel(x)
    return s, (A, B, C)
    
parallel_gflops = [benchmark_square_matmul_tvm(n, parallel) for n in sizes]

plot_gflops(sizes, [np_gflops, default_gflops, reorder_gflops, parallel_gflops], 
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
