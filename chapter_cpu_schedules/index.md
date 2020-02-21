# Operator Optimizations on CPUs
:label:`ch_cpu_schedules`

In the past three chapters we mainly focus on the functionality of operators, namely, how to implement the operators to function correctly in TVM. However, getting right results out is not sufficient. Operators could perform poorly even if they execute correctly.

Starting from this chapter, we will talk about the performance optimization to the operators. Specifically, we will work on operator optimizations on CPUs in this chapter, and move on to GPUs in the next chapter.

```toc
:maxdepth: 2
:numbered:

arch
call_overhead
vector_add
broadcast_add
matmul
block_matmul
conv
packed_conv
depthwise_conv
```
