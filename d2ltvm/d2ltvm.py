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
from mxnet import np as mp, npx as mpx
mpx.set_np()


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


# Defined in file: ./chapter_cpu_schedule/call_overhead.md
def bench_workload(workload):
    """Benchmarka a workload
    
    workload - must accept a num_repeat argument and return the total runtime
    """
    workload(1)  # warmup
    time = workload(1)  # the time to run once
    if time > 1: return time
    # The number of repeats to measure at least 1 second.
    num_repeats = max(int(1.0 / time), 5)
    return workload(num_repeats) / num_repeats


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
    timer = timeit.Timer(
        setup='import numpy as np\n'
        'n = ' + str(n) + '\n'
        'x = np.random.normal(size=(n, n)).astype(np.float32)\n'
        'y = np.random.normal(size=(n, n)).astype(np.float32)\n'
        'z = np.empty_like(x)\n',
        stmt = 'np.dot(x, y, out=z);')
    # Estimate the #repeat to run for 1 second
    time = timer.timeit(1)
    nrepeat = max(int(1.0/time), 5) 
    time = timer.timeit(nrepeat) 
    return 2 * n**3 / time / 1e9 * nrepeat


# Defined in file: ./chapter_cpu_schedule/matmul.md
def square_matmul_default(n):
    """Return the computing expression of square matrix multiplication with
    the default schedule.
    """
    k = tvm.reduce_axis((0, n), name='k')
    A = tvm.placeholder((n, n), name='A')
    B = tvm.placeholder((n, n), name='B')
    C = tvm.compute(
        (n, n), lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k), name='C')
    return tvm.create_schedule(C.op), (A, B, C)


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
    


# Defined in file: ./chapter_cpu_schedule/basic_conv.md
def conv_out_size(n, k, p, s):
    """Compute the output size by given input size n, 
    kernel size k, padding p, and stride s"""
    return (n - k + 2 * p)//s + 1


# Defined in file: ./chapter_cpu_schedule/basic_conv.md
def conv_gflop(oc, ic, n, k, p, s):
    """Compute the #floating points in a convolution by given
    output channels oc, input channels ic, input size n, kernela size k, 
    padding p and stride s
    """
    on = conv_out_size(n, k, p, s)
    return 2 * oc * ic * on * on * k * k / 1e9


# Defined in file: ./chapter_cpu_schedule/basic_conv.md
def get_data(shape, func, target=tvm.cpu()):
    if func.__name__ in ('normal', 'uniform'):
        data = func(size=shape)
    else:
        data = func(shape=shape)
    if hasattr(data, 'asnumpy'):
        data = data.asnumpy()
    return tvm.nd.array(data.astype('float32'), target)


# Defined in file: ./chapter_cpu_schedule/basic_conv.md
def get_conv_data_mxnet(oc, ic, n, k, p, s):
    mpx.random.seed(0)
    data = mp.random.normal(size=(1, ic, n, n))
    weight = mp.random.normal(size=(oc, ic, k, k))
    bias = mp.zeros((oc,))
    on = conv_out_size(n, k, p, s)
    out = mp.empty((1, oc, on, on))
    # Wait data are generated to make later benchmarking accurate
    mpx.waitall()  
    return data, weight, bias, out


# Defined in file: ./chapter_cpu_schedule/basic_conv.md
def benchmark_mod_tvm(mod, args, target):
    # Estimate the #repeat to run for 1 second, with at least 5 runs
    start = time.time()
    mod(*args)
    nrepeat = int(max(1.0/(time.time() - start), 5))
    ctx = tvm.context(target, 0)
    timer = mod.time_evaluator(mod.entry_name, ctx=ctx, number=nrepeat)
    return timer(*args).mean


# Defined in file: ./chapter_cpu_schedule/basic_conv.md
def benchmark_conv_mxnet(oc, ic, n, k, p, s):
    timer = timeit.Timer(
        setup='import d2ltvm\n'
        'from mxnet import npx\n'
        'oc, ic, n, k, p, s = %d, %d, %d, %d, %d, %d\n'
        'data, weight, bias, out = d2ltvm.get_conv_data_mxnet(\n'
        '    oc, ic, n, k, p, s)'%(oc, ic, n, k, p, s),
        stmt='npx.convolution(data, weight, bias, kernel=(k,k), pad=(p,p),\n' 
        '    stride=(s,s), num_filter=oc, out=out); out.wait_to_read()\n')
    # Estimate the #repeat to run for 1 second, with at least 5 runs
    nrepeat = max(int(1.0/timer.timeit(1)), 3)
    time = timer.timeit(nrepeat)
    return conv_gflop(oc, ic, n, k, p, s) / time  * nrepeat


