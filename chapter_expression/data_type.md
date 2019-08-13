# Data Types

The `tvm_vector_add` module we created in :numref:`ch_vector_add` only accepts `float32` arguments. Let's extend it to other data types in this section.


## Create with a Specify Data Type

To use a different data type, we need to specify it explicitly when creating the placeholders. We put the codes for vector addition in :numref:`ch_vector_add` into the function `tvm_vector_add`, which accepts an argument `dtype` for data type. We pass `dtype` to `tvm.placeholder` to create `A` and `B` with the given data type. 

```{.python .input}
import tvm
import d2ltvm
import numpy as np

n = 100

def tvm_vector_add(dtype):
    A = tvm.placeholder((n,), name='a', dtype=dtype)
    B = tvm.placeholder((n,), name='b', dtype=dtype)
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='c')
    s = tvm.create_schedule(C.op)
    return tvm.build(s, [A, B, C])
```

Let's compile an module that accepts `int32` arguments. 

```{.python .input}
mod = tvm_vector_add('int32')
```

We also save the test codes in :numref:`ch_vector_add` in the function `test_mod` and verity the results.

```{.python .input}
def test_mod(mod, dtype):    
    a, b = (np.random.normal(size=100).astype(dtype) for _ in range(2))
    c = np.empty(100, dtype=dtype)
    _, _, c = d2ltvm.eval_mod(mod, a, b, c)
    print('datatype of c', c.dtype)
    np.testing.assert_equal(c, a + b)
    
test_mod(mod, 'int32')
```

Let's try other data types as well

```{.python .input}
for dtype in ['float16', 'float64', 'int8','int16', 'int64']:
    mod = tvm_vector_add(dtype)
    test_mod(mod, dtype)
```

## Convert Data Types

usage of `astype`. TODO. 

```{.python .input}
# A = tvm.placeholder((n,), name='a').astype(dtype) # FIXME
```

## Summary

- We can specify the data type by `dtype` when creating placeholders.
- convert by `astype`.
