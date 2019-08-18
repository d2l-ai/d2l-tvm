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
def eval_mod(mod, *args):
    tvm_args = [
        tvm.nd.array(x) if isinstance(x, np.ndarray) else x for x in args]
    mod(*tvm_args)
    for x, y in zip(args, tvm_args):
        if isinstance(x, np.ndarray):
            np.copyto(x, y.asnumpy())

# Defined in file: ./chapter_schedule/cpu/vector_add.md
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

