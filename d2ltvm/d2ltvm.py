# This file is generated automatically through:
#    d2lbook build lib
# Don't edit it directly

import sys
d2ltvm = sys.modules[__name__]

# Defined in file: ./chapter_expression/vector_add.md
def eval_mod(mod, *args):
    tvm_args = [tvm.nd.array(arr) for arr in args]
    mod(*tvm_args)
    return [arr.asnumpy() for arr in tvm_args]

