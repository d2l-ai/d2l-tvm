# Improve Cache Efficiency by Blocking 


```{.python .input}
%matplotlib inline
import tvm
import numpy as np
import d2ltvm 

np.show_config()
```

```{.python .input}
sizes = 2**np.arange(5, 12, 1)
np_gflops = [d2ltvm.benchmark_square_matmul_np(n) for n in sizes]
```

```{.python .input}
def yy(n):
    bn = 32
    k = tvm.reduce_axis((0, n), name='k')
    A = tvm.placeholder((n, n), name='A')
    B = tvm.placeholder((n, n), name='B')

    packedB = tvm.compute((n / bn, n, bn), lambda x, y, z: B[y, x * bn + z], name='packedB')
    C = tvm.compute((n, n),
                    lambda x, y: tvm.sum(A[x, k] * packedB[y // bn, k, tvm.indexmod(y, bn)], axis=k),
                    name = 'C')


    s = tvm.create_schedule(C.op)

    CC = s.cache_write(C, 'global')

    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

    s[CC].compute_at(s[C], yo)

    xc, yc = s[CC].op.axis

    k, = s[CC].op.reduce_axis
    ko, ki = s[CC].split(k, factor=2)
    s[CC].reorder(ko, xc, ki, yc)
    s[CC].unroll(ki)
    s[CC].vectorize(yc)

    # parallel
    s[C].parallel(xo)

    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)
    return s, (A, B, C)


block_gflops = [d2ltvm.benchmark_square_matmul_tvm(n, yy) for n in sizes]

d2ltvm.plot_gflops(sizes, [np_gflops, block_gflops], 
            ['numpy', 'default'])
```
