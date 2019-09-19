Dive into Deep Learning Compiler
================================

**Working in progress. Please check back later.**

This project is for readers who are interested in high-performance
implementation of their programs for scientific computing, but may only know
NumPy before. We are not aiming to replace the tutorials available at tvm.ai,
but for more easy-to-read tutorials with a consistent theme among each
section.

```toc
:maxdepth: 1

chapter_install/install
```

There are two parts of TVM that related to us to accelerate deep learning
workloads. One is writing highly efficient operators, such as matrix product, on various hardware. The other part is optimizing the whole program, such as fusing small operators into a bigger one. This book focus on the first part, while leaving the second part for future works.

To write an operator, we first define its computation expression. Like CUDA, we assume the workload can be executed in a data parallelism style. Here we only need to define how each element in the output is computed based on the inputs. We will present how to define the computation using the DSL (domain specific language) provided in by TVM.

```toc
:maxdepth: 2
:numbered:

chapter_expression/index

```

Next, we need to define how an expression is executed. If the output is a 10-by-10 matrix, then we need to run the expression 100 times. Naively running them sequentially will leads to poor performance. We need to fully utilize the machine resources for better performance, including SIMD (Single instruction, multiple data), multi-cores, and memory cache. We will demonstrate typical scheduling optimizations on multiple hardware.

```toc
:maxdepth: 2
:numbered:

chapter_cpu_schedule/index
chapter_gpu_schedule/index
```
