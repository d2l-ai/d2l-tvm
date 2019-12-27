# CPU Architecture
:label:`ch_cpu_arch`


In this section, we will do a brief introduction to the system components that are important for the performance of deep learning and scientific computing on CPUs. For a more comprehensive survey, we recommend [this classic textbook](https://www.amazon.com/Computer-Architecture-Quantitative-John-Hennessy/dp/012383872X). We assume the readers knowing the basic system concepts such as clock rate (frequency), CPU cycle, and cache.

## Arithmetic Units

A typical general-purpose CPU has hardware units to perform arithmetics on integers (called [ALU](https://en.wikipedia.org/wiki/Arithmetic_logic_unit)) and floating-points (called [FPU](https://en.wikipedia.org/wiki/Floating-point_unit)). The performance of various data types depends on the hardware. Let's first check the CPU model we are using.

```{.python .input  n=1}
# The following code runs on Linux
!cat /proc/cpuinfo | grep "model name" | head -1
```

Now check the performance of a matrix multiplication of different data types.

```{.python .input  n=6}
import numpy as np

def benchmark(dtype):
    x = np.random.normal(size=(1000, 1000)).astype(dtype)
    %timeit np.dot(x, x)

benchmark('float32')
benchmark('float64')
benchmark('int32')
benchmark('int64')
```

As can be seen, 32-bit floating-point (float32) is 2x faster than 64-bit floating-point (float64). The integer performance is way more slower and there is no much difference between 32-bit integer (int32) and 64-int integer. We will get back to the understand more about these numbers later.

Some operators, however, could be significantly slower than the multiplication and addition `a += b * c` used in matrix multiplication. For example, CPU may need hundreds of cycles to computing transcendental functions such as `exp`. You can see that even 1000 times fewer operations is needed for `np.exp(x)` than `np.dot(x, x)`, the former one takes longer time.

```{.python .input  n=14}
x = np.random.normal(size=(1000, 1000)).astype('float32')
%timeit np.exp(x)
```

## Parallel Execution

The CPU frequency increased rapidly until the beginning of the 21st century. In 2003, Intel released a [Pentium 4](https://en.wikipedia.org/wiki/Pentium_4) CPU with up to 3.8 GHz clock rate. If we check our CPU clock rate,

```{.python .input}
# The following code runs on Linux
!lscpu | grep MHz
```

we can see that it has a lower clock rate compared to the product in 2003, but it might be 100x faster than the Pentium 4 CPU. One secret source is that new CPU models explore a lot more in the territory of parallel execution. Next we briefly discuss two typical parallelizations.

![Single core vs. single core with SIMD vs. multi-core with SIMD.](../img/cpu_parallel_arch.svg)
:label:`fig_cpu_parallel_arch`

### SIMD

Single instruction, multiple data ([SIMD](https://en.wikipedia.org/wiki/SIMD)), refers to processing multiple elements with the same instruction simultaneously. :numref:`fig_cpu_parallel_arch` illustrates this architecture. In a normal CPU core, there is an instruction fetching and decoding unit. It runs an instruction on the processing unit (PU), e.g. ALU or FPU, to process one element, e.g. float32, each time. With SIMD, we have multiple PUs instead of one. In each time, the fetch-and-decode unit submit the same instruction to every PU to execute simultaneously. If there are $n$ PUs, then we can process $n$ element each time.

Popular SIMD instruction sets include Intel's [SSE](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions) and [AVX](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions), ARM's [Neon](https://en.wikipedia.org/wiki/ARM_architecture#Advanced_SIMD_(NEON)) and AMD's [3DNow!](https://en.wikipedia.org/wiki/3DNow!). Let's check which sets our CPU supports.

```{.python .input}
# The following code runs on Linux
!cat /proc/cpuinfo | grep "flags" | head -1
```

As can be seen, the most powerful SIMD instruction set supported is AVX-512, which
extends AVX to support executing SIMD on 512-bit width data, e.g. it is able to perform 16 float32 operations or 8
float64 operations each time.

### Multi-cores

SIMD improves the performance of a single core, another way is adding multiple
cores to a single CPU processor. numref:`fig_cpu_parallel_arch` shows two CPU
cores, each of which has 2 PUs. 

It looks like that our CPU has 16 cores.

```{.python .input}
# The following code runs on Linux
!cat /proc/cpuinfo | grep "model name" | wc -l
```

But note that modern Intel CPUs normally has 
[hyper-threading](https://en.wikipedia.org/wiki/Hyper-threading) which runs 2 hardware
threads per core. By hyper-threading, each core is presented
as 2 logical cores to the operating system. So even the system shows there are 16
cores, physically our CPU only has 8 cores.

Having two threads sharing the resource of the same core may increase the total throughput but at the expense of increasing the overall latency.
In addition the effect of hyper-threading is very much dependent on the application.
Therefore, it is not generally recommended to leverage hyper-threading in the deep learning workloads.
Later on in the book, you'll see that we only launch 8 threads even if our CPU presents 16 cores.

### Performance

We often use floating point operations per second ([FLOPS](https://en.wikipedia.org/wiki/FLOPS)) to measure the performance of a hardware platform or an executable program.
The theoretical peak performance of a single CPU can be computed by

`#physical_cores * #cycles_per_second * #instructions_per_cycle * #operations_per_instruction`

where `#instructions_per_cycle` is also called the SIMD width.

For the CPU we are using, it has 8 physical cores, the max clock rate (i.e. `#cycles_per_second`) is $2.5\times 10^9$, the AVX-512 computes 16 float32 instructions per cycle, the [FMA](https://en.wikipedia.org/wiki/FMA_instruction_set) instruction set in AVX-512 compute `a += b * c` each time, which contains 2 operations. Therefore, the GFLOPS (gigaFLOPS) for single precision (float32) is

```{.python .input}
2.5 * 8 * 16 * 2
```

You can modify the above code based on your system information to calculate your CPU peak performance.

Matrix multiplication (*matmul*) is a good benchmark workload for the peak performance, which has $2\times n^3$ operations in total if all matrices are in shape $[n, n]$. After executing a *matmul*, we can get its (G)FLOPS by dividing its total operations using the averaged executing time. As can be seen, the measured GFLOPS is close to the peak performance (~90% of peak).

```{.python .input}
x = np.random.normal(size=(1000, 1000)).astype('float32')
res = %timeit -o -q np.dot(x, x)
2 * 1000**3 / res.average / 1e9
```

## Memory Subsystem

Another component which significantly impacts the performance is the memory subsystem. The memory size is one of the key specifications of a system. The machine we are using has 240 GB memory.

```{.python .input}
# The following code runs on Linux
!cat /proc/meminfo | grep MemTotal
```

The memory bandwidth, on the other hand, is less noticed but equally important. We can use the
[mbw](http://manpages.ubuntu.com/manpages/xenial/man1/mbw.1.html) tool to test
the bandwidth.

```{.python .input}
# The following code runs on Linux
!mbw 256 | grep AVG | grep MEMCPY
```

Note that our CPU can execute $640\times 10^9$ operations on float32 numbers per second. This
requires the bandwidth to be at least $640\times 4=2560$ GB/s, which is significantly
larger than the measured bandwidth. CPU uses caches to fill
in this big bandwidth gap. Let's check the caches our CPU has.

```{.python .input}
# The following code runs on Linux
!lscpu | grep cache
```

As can be seen, there are three levels of caches: L1, L2 and L3 (or LLC, Last Level Cache). The L1 cache has 32KB for instructions and 32KB for data. The L2 cache is 32x larger. The L3 cache is way more larger, but it is still thousands times smaller than the main memory. The benefits of caches are significantly improved access latency and bandwidth. Typically on modern CPUs,
the latency to access L1 cache is less than 1 ns, the L2 cache's latency is around 7 ns, and the L3 cache is slower, with a latency about 20 ns, while still faster than the main memory's 100 ns latency.


![The layout of main memory and caches.](../img/cpu_memory.svg)
:label:`fig_cpu_memory`

A brief memory subsystem layout is illustrated in :numref:`fig_cpu_memory`.
L1 and L2 caches are exclusive to each CPU core, and L3 cache is shared across the cores of the same CPU processor
To processing on some data, a CPU will first check if the data exist at L1 cache, if not check L2 cache, if not check L3 cache, if not go to the main memory to retrieve the data and bring it all the way through L3 cache, L2 cache, and L1 cache, finally to the CPU registers.
This looks very expensive but luckily in practice, the programs have the [data locality patterns](https://en.wikipedia.org/wiki/Locality_of_reference) which will accelerate the data retrieving procedure. There are two types of locality: temporal locality and spatial locality.
Temporal locality means that the data we just used usually would be used in the near future so that they may be still in cache. Spatial locality means that the adjacent data of the ones we just used are likely to be used in the near future. As the system always brings a block of values to the cache each time (see the concept of [cache lines](https://en.wikipedia.org/wiki/CPU_cache#CACHE-LINES)), those adjacent data may be still in cache when referenced to.
Leveraging the advantage brought by data locality is one of the most important performance optimization principles we will describe in detail later.

## Summary

- CPUs have dedicated units to handle computations on various data types. A CPU's peak performance is determined by the clock rate, the number of cores, and the instruction sets.
- CPUs use multi-level caches to bridge the gap between CPU computational power and main memory bandwidth.
- An efficient program should be effectively parallelized and access data with good temporal and spatial localities.
