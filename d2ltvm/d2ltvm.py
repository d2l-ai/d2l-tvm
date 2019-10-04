# This file is generated automatically through:
#    d2lbook build lib
# Don't edit it directly

import sys
d2ltvm = sys.modules[__name__]

# Defined in file: ./chapter_install/install.md
import tvm
import time
import timeit
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


# Defined in file: ./chapter_cpu_schedule/matmul.md
def benchmark_square_matmul_np(n):
    nrepeat = max(int(1e9/n**3), 5)
    time = timeit.timeit(
        setup='import numpy as np\n'
        'n = ' + str(n) + '\n'
        'x = np.random.normal(size=(n, n)).astype(np.float32)\n'
        'y = np.random.normal(size=(n, n)).astype(np.float32)\n'
        'z = np.empty_like(x)\n',
        stmt = 'np.dot(x, y, out=z);',
        number = nrepeat)
    return 2 * n**3 / time / 1e9 * nrepeat


# Defined in file: ./chapter_cpu_schedule/matmul.md
def benchmark_square_matmul_tvm(n, generator, target='llvm -mcpu=core-avx2'):
    # Compile
    s, [A, B, C] = generator(int(n))
    mod = tvm.build(s, [A, B, C], target=target)
    # Prepare inputs and outputs
    x = np.random.normal(size=(n, n)).astype(np.float32)
    y = np.random.normal(size=(n, n)).astype(np.float32)
    z = np.empty_like(x)
    ctx = tvm.context(target, 0)
    x, y, z = tvm.nd.array(x, ctx), tvm.nd.array(y, ctx), tvm.nd.array(z, ctx)
    # Estimate the #runs to roughly benchmark for 1 second
    start = time.time()
    mod(x, y, z)
    nrepeat = int(max(1.0/(time.time() - start), 1))
    timer = mod.time_evaluator(mod.entry_name, ctx=ctx, number=nrepeat)
    return 2 * n**3 / timer(x, y, z).mean / 1e9


# Defined in file: ./chapter_cpu_schedule/matmul.md
def plot_gflops(sizes, gflops, legend):
    d2ltvm.plot(sizes, gflops, xlabel='Size', ylabel='GFLOPS', 
             xscale='log', yscale='log', 
             legend=legend, fmts=['--']*(len(gflops)-1)+['-'])
    


