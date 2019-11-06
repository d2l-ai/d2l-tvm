Dive into Deep Learning Compiler
================================

**Working in progress. Check our [roadmap](http://bit.ly/2NQ7gh3) for more details**

This project is for readers who are interested in high-performance
implementation of their programs for scientific computing, especially deep
learning models, but may haven't got their hands dirty yet. We assume
readers have a minimal background, they may only use NumPy before. It means that
we will explain things from scratch and introduce relative background when needed. Experienced
readers, however, should also find the contents useful.

We roughly classify contents into three major parts. In the first part, we will
introduce how to implement and optimize operators, such as matrix multiplication
and convolution, for various hardware. This is the basic component for
scientific computing. In the second part, we will show how to convert neural
network models from various deep learning frameworks and further optimize them
in the program level. The last part we will address how to deploy the optimized
program into various environment such as mobile phones.


```toc
:maxdepth: 2
:numbered:

chapter_getting_started/index
chapter_expressions/index
chapter_common_operators/index
chapter_cpu_schedules/index
chapter_gpu_schedules/index
chapter_neural_networks/index
chapter_deployment/index
```


```toc
:maxdepth: 1

chapter_references/zreferences
```
