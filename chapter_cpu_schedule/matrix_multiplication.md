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
    
#benchmark_square_matmul(np_dot, 1024)
```

Now we benchmark the performance on various input sizes as our baseline.

```{.python .input  n=3}
def benchmark(func, constructor=None):
    gflops, sizes = [], (2**np.arange(5, 12, 1)).tolist()
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
def matrix_product(n):
    k = tvm.reduce_axis((0, n), name='k')
    A = tvm.placeholder((n, n), name='A')
    B = tvm.placeholder((n, n), name='B')
    C = tvm.compute(
        (n, n), lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k), name='C')
    return (A, B, C)
```

The default schedule follows the computation illustrated in :numref:`fig_matmul_default`.

```{.python .input  n=22}
def benchmark_tvm(schedule_updater=None):
    def func(n):
        A, B, C = matrix_product(n)
        s = tvm.create_schedule(C.op)
        if schedule_updater is not None: 
            schedule_updater(s, C)
        mod = tvm.build(s, [A, B, C], target='llvm -mcpu=core-avx2')
        return mod
    return func

_, default_gflops = benchmark(benchmark_tvm(), tvm.nd.array)
```

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
    
_, reorder_gflops = benchmark(benchmark_tvm(reorder), tvm.nd.array)    
```

We can see that the reordering significantly improves the performance compared to the default schedule.

```{.python .input  n=9}
plot([np_gflops, default_gflops, reorder_gflops], 
     ['numpy', 'default', 'reorder'])
```

## Parallelization

In the outermost for-loop, each time we compute the results of a row in $C$. Each row can be computed in parallel, so we can make the schedule be parallelized on axis `x`.

```{.python .input  n=10}
import os 
os.environ["TVM_NUM_THREADS"] = '16'

def parallel(s, C):
    (x, y), (k,) = C.op.axis, C.op.reduce_axis
    s[C].reorder(x, k, y)
    s[C].parallel(x)
    
_, parallel_gflops = benchmark(benchmark_tvm(parallel), tvm.nd.array)    
```

Parallelization improves the performance again. But we can see that there is still a gap compared to NumPy on large matrices.

```{.python .input  n=11}
plot([np_gflops, default_gflops, reorder_gflops, parallel_gflops], 
     ['numpy', 'default', 'reorder', '+ parallel'])
```

## Block Tiling

Another popular way to improve the memory localization is block tiling. The idea is that a block of $C$, e.g. `C[x:x+tx, y:y+ty]` by the NumPy notation, can be computed by the according rows of $A$ and columns of $B$. That is


``C[x:x+tx, y:y+ty] = np.dot(A[x:x+tx,:], B[:,y:y+ty])``

We can further decompose this matrix multiplication into multiple small ones

``C[x:x+tx, y:y+ty] = sum(np.dot(A[x:x+tx,k:k+tk], B[k:k+tk,y:y+ty]) for k in range(0,n,tk))``

It is also illustrate in :numref:`fig_matmul_block`. If we choose proper tiling sizes `tx`, `ty` and `tk` to fit the block matrices of $A$, $B$ and $C$ into the cache, then the reduced cache miss rate will improve the performance. 

![](../img/matmul_block.svg)
:label:`fig_matmul_block`

Let's implement this idea. We first split each axis into two by the `split` method with the specified tilling sizes, which are tunable hyper-parameters. Then we reorder the axis into two parts, each part has three for-loops. The inner part performs matrix multiplication on two submatrices, while the outer part iterates over all submatrices. Similarly as before, we parallelize the workloads in the outermost for loop. In addition, we hint the compiler to use vectorized instructions, such as `avx`, for the innermost for loop, and unroll the other two loops in the inner part.

```{.python .input  n=39}
# The tiling sizes
tx, ty, tk = 8, 8, 8

def block(s, C):    
    (x, y), (k,) = C.op.axis, C.op.reduce_axis

    xo, xi = s[C].split(x, tx)
    yo, yi = s[C].split(y, ty)
    ko, ki = s[C].split(k, tk)

    s[C].reorder(xo, ko, yo, xi, ki, yi)
    s[C].vectorize(yi)
    s[C].unroll(xi)
    s[C].unroll(ki)
    s[C].parallel(xo)

    
_, block_gflops = benchmark(benchmark_tvm(block), tvm.nd.array)    

```

As you can seen, block tiling not always improves the performance. There are three reasons. First, re-constructing the matrices into blocks has overhead. Second, the granularity of the parallelized workloads is `tx` times more coarse, which may decrease performance when `n` is relatively small. Third, the tiling sizes may not be optimal.

```{.python .input  n=40}
plot([np_gflops, default_gflops, reorder_gflops, parallel_gflops, block_gflops], 
     ['numpy', 'default', 'reorder', '+ parallel', '+ block'])
```

Finally, let's print the GFLOPS for $n=1024$ and improve it in the next section.

```{.python .input}
n = 1024
mod = benchmark_tvm(block)(n)
benchmark_square_matmul(mod, n, tvm.nd.array)

```

## Summary

1. Reordering the for-loops in matrix multiplication properly improves the performance. 
2. Parallelization is also important.
3. Block tiling may further improve the performance.
