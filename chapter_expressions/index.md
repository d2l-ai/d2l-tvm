# Expressions for Operators

We start from operators. As you may know, operators are the building blocks of 
neural network models. A deep neural network can be expressed as a Directed Acyclic Graph (DAG),
which consists of nodes being the operators and edges being the data dependency between nodes.
Being able to execute the operators efficiently is of course a necessity to high-performance
neural network model execution.

In :numref:`ch_vector_add` you have seen how to build the vector addition
expression in TVM. This chapter covers more concepts in TVM to construct
expressions. Specifically, you'll learn about data types, shapes, indexing,
reduction and control flow, based on which you'll be able to construct 
operators in the next chapter.

```toc
:maxdepth: 2

data_types
shapes
index_shape_expressions
reductions
if_then_else
all_any
```

