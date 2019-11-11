# Dive into Deep Learning Compiler

**Working in progress**. Check our [roadmap](http://bit.ly/2NQ7gh3) for more details.

This project is for readers who are interested in high-performance
implementation of their programs ultilizing deep learning techniques, especially model inference,
but may not have got their hands dirty yet. We assume
readers have a minimal background of only having experience on NumPy before. With this in mind,
we will explain things from scratch and introduce relative background when needed. Experienced
readers, however, should also find the contents useful.


We roughly classify contents into three major parts. In the first part, we will
introduce how to implement and optimize operators, such as matrix multiplication
and convolution, for various hardware platforms. This is the basic component for
deep learning as well as scientific computing in general.
In the second part, we will show how to convert neural
network models from various deep learning frameworks and further optimize them
in the program level. The last part we will address how to deploy the optimized
program into various environment such as mobile phones.
In addition, at the end of the book,
we plan to cover some latest advance of the deep learning compiler domain.


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

## [Discuss](https://discuss.tvm.ai/t/d2l-tvm-a-tvm-introduction-book/4305)
