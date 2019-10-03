# This file is generated automatically through:
#    d2lbook build lib
# Don't edit it directly

import sys
d2ltvm = sys.modules[__name__]

# Defined in file: ./chapter_install/install.md
import tvm
import numpy as np
from matplotlib import pyplot as plt
from IPython import display


# Defined in file: ./chapter_expression/vector_add.md
def eval_mod(mod, *inputs, out):
    """Evaluate a TVM module, and save results in out.
    """
    # Convert all numpy arrays to tvm arrays
    tvm_args = [tvm.nd.array(x) if isinstance(x, np.ndarray) 
                else x for x in inputs + (out,)]
    mod(*tvm_args)
    # If out is a tvm array, then its value has already been inplaced. 
    # Otherwise, explicitly copy the results. 
    if isinstance(out, np.ndarray):
        np.copyto(out, tvm_args[-1].asnumpy())


# Defined in file: ./chapter_cpu_schedule/vector_add.md
def plot(X, Y, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear', fmts=None,
         figsize=(6, 4)):
    """Plot multiple lines"""
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize
    axes = plt.gca()
    X, Y = np.array(X), np.array(Y)
    if X.shape != Y.shape: X = [X] * len(Y)
    if not fmts: fmts = ['-'] * len(X)
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend: axes.legend(legend)
    axes.grid()


# Defined in file: ./chapter_cpu_schedule/matrix_multiplication.md
def square_matmul(n):
    """Return the computing expression of square matrix multiplication. 
    """
    k = tvm.reduce_axis((0, n), name='k')
    A = tvm.placeholder((n, n), name='A')
    B = tvm.placeholder((n, n), name='B')
    C = tvm.compute(
        (n, n), lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k), name='C')
    return (A, B, C)


# Defined in file: ./chapter_cpu_schedule/matrix_multiplication.md
def square_matmul_module(schedule_updater=None, 
                         target='llvm -mcpu=core-avx2'):
    """Returns a function that accepts the input size n to return
    a TVM module. 
    """
    def func(n):
        A, B, C = square_matmul(n)
        s = tvm.create_schedule(C.op)
        if schedule_updater is not None: 
            schedule_updater(s, C)
        mod = tvm.build(s, [A, B, C], target=target)
        return mod
    return func


