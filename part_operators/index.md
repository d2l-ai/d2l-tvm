# Operators

If you used NumPy before, you will know that NumPy provides close to 1,000
operators and the main part of your program may be based on these
operators. NumPy's oprators are well-optimized, and therefore often outperforms
the same program that are purely implemented by Python.

In this part, we will show how to implement various operators from scratch and
then optimize their performance for both CPU and GPU. We first describe the TVM
IR, which is a domain specif language for Python, and then show the
implementations of the commonly used operators. Next we discuss optimization
technologies to better utilize the hardware resources such as SIMD (Single
instruction, multiple data) and multi-cores.

```toc
:maxdepth: 2
:numbered:

expressions/index
common_operators/index
cpu_schedules/index
gpu_schedules/index
```
