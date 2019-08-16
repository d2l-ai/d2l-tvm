# This file is generated automatically through:
#    d2lbook build lib
# Don't edit it directly

import sys
d2ltvm = sys.modules[__name__]

# Defined in file: ./chapter_install/install.md
import tvm
import numpy as np

# Defined in file: ./chapter_expression/vector_add.md
def eval_mod(mod, *args):
    tvm_args = [
        tvm.nd.array(x) if isinstance(x, np.ndarray) else x for x in args]
    mod(*tvm_args)
    for x, y in zip(args, tvm_args):
        if isinstance(x, np.ndarray):
            np.copyto(x, y.asnumpy())

