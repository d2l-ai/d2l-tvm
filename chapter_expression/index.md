# Computing

In this part, we will describe how to implement various operators in TVM. We know that parallelization is one of the key technologies to improve the performance of scientific computing. Data parallelism is the widely used mechanism in parallel computing. As a result, TVM adopts the SPMD (single program, multiple data) scheme. Similar to CUDA, we don't need to write the whole program but define how a single output element should be computed.

For example, to define the vector addition $c=a+b$, in Python we can write

```python
for i in range(len(c)):
    c[i] = a[i] + b[i]
```

While in TVM, we only need to specify how `c[i]` is computed. This expression will be then applied to all possible `i`s automatically. Besides the benefit of a more concise program, it allows more flexible ways to schedule the computing for better performance later on.

```.toc
:maxdepth: 2

vector_add
data_type
shape
index_shape_expression
reduce
if_else
scan
```
