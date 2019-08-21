# Function Call Overhead

We are starting to benchmark various schedules since this chapter. Before diving into various running time numbers, we need to be aware of the overhead of issuing a function call in Python. It's well known that Python is not the fastest language on earth. Prototyping with Python is fast, but we also need to pay the cost that Python does smart things for us under the hook. In this section, we will investigate the overhead to call a function in Python, and demonstrate its impact to our later benchmarking results. 

We will use the magic build-in function `%timeit` to benchmark our codes. Let's first check how it looks.

```{.python .input  n=22}
%timeit a = 1
```

As we can see, `%timeit` repeat 7 runs, in each run it iterates our function `a=1` by $10^7$ times. The average cost of executing `a=1` is $13\times 10^{-9}$ second. The standard deviation is small, so this results are trustable. Though we can reduce its latency below a nanosecond in a fast language such as C, we know that `%timeit` is able to provide results in nanosecond precision. 

## The Not Negligible Overhead

Now let's define a benchmark function to copy a vector into another vector. We pass `-o` and `-q` to `timeit` to obtain the output and disable printing.

```{.python .input  n=45}
import numpy as np
from matplotlib import pyplot as plt
from IPython import display

def benchmark(func, n_start, n_end, n_stride=1):
    avg_times, sizes = [], (2**np.arange(n_start, n_end, n_stride))
    np.random.seed(0)
    for size in sizes:
        avg_times.append(func(size))
    return sizes, np.array(avg_times)

def np_copy(size):
    x = np.random.normal(size=size).astype('float32')
    y = np.empty_like(x)
    res = %timeit -o -q np.copyto(y, x)
    return res.average
```

Now run the benchmark and plot the throughput versus vector length.

```{.python .input  n=49}
sizes, times = benchmark(np_copy, 5, 30, 4)

display.set_matplotlib_formats('svg')
plt.loglog(sizes, sizes*4/times/1e9)
plt.xlabel('Vector length')
plt.ylabel('Throughput (GB/sec)')
plt.grid()
```

We can see that the throughput increases and then becomes stable. It's not as expected. When the vector length is small, more likely we can fit it into cache, and therefore the throughput should be high. To examine the reason, let's simply draw the execution time versus vector length.

```{.python .input  n=52}
plt.loglog(sizes, times)
plt.xlabel('Vector length')
plt.ylabel('Time (sec)')
plt.grid()
```

We can see that when the vector length is smaller than $10^3$, the execution time is almost flat. It is dominated by the function call overhead. The overhead includes any argument preprocessing in the Python function, evoking the foreign function interface, and other [Python backend overhead](https://jakevdp.github.io/blog/2014/05/09/why-python-is-slow/). Therefore, benchmarking too small workloads is not quite meaningful. 

## Overhead of NumPy, TVM and MXNet

During this book, we will benchmark various operators in Numpy, TVM and MXNet. Let's examine their function call overhead. We could roughly estimate it by executing some small workloads. Let first check NumPy. 

```{.python .input}
_, times = benchmark(np_copy, 1, 8)
print('NumPy call overhead: %.1f microsecond'% (times.mean()*1e6,))
```

The overhead of TVM is around 5x higher.

```{.python .input}
import tvm 

def tvm_copy(size):
    x = np.random.normal(size=size).astype('float32')
    y = np.empty_like(x)
    x, y = tvm.nd.array(x), tvm.nd.array(y)
    res = %timeit -o -q x.copyto(y)
    return res.average

_, times = benchmark(tvm_copy, 1, 8)
print('TVM call overhead: %.1f microsecond'% (times.mean()*1e6,))
```

MXNet has the highest overhead, it's even 8x times higher than TVM. The reason might due to MXNet uses `ctypes` while TVM is compiled with `cython`.

```{.python .input}
import mxnet as mx 

def mx_copy(size):
    x = np.random.normal(size=size).astype('float32')
    y = np.empty_like(x)
    x, y = mx.nd.array(x), mx.nd.array(y)
    res = %timeit -o -q x.copyto(y)
    return res.average

_, times = benchmark(mx_copy, 1, 8)
print('MXNet call overhead: %.1f microsecond'% (times.mean()*1e6,))
```

## Summary

- The function call overhead might takes several microsecond. Benchmarking too small functions in Python is meaningless. 
