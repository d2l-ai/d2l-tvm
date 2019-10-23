# Hyperparameter Turning for Matrix Multiplication
:label:`ch_auto_matmul_cpu`

In the last part of :numref:`ch_matmul_cpu` we introduced block tiling, whose performance strongly relies on the tiling sizes. The optimal values often depend on the matrix sizes and the CPU specification. In this chapter, we will introduce how to search a good set of hyperparameters.

```{.python .input  n=1}
import tvm
from tvm import autotvm
import d2ltvm as d2l
import logging
import sys
import os

os.environ["TVM_NUM_THREADS"] = '16'

# Show detailed tuning log
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
```

The basic idea is that, instead of giving concrete values to the hyperparameters `tx`, `ty` and `tk`, we will specify multiple value candidates for each hyperparameter. Then we repeatedly sample a set of values from all candidates each time to benchmark its performance. The one achieved the best performance is then picked. 

## Create a template 

There are two key components, one is how to pick the candidates, and the other one is the sampling methods. For the first one, we heuristically pick a list of values for each hyperparameter. It can be specified by creating a knob in `autotvm`. Then we pass the `.val` attribute, which denotes by a value in list, to the `split` function. The rest part is almost same as before, but note that we put the function in the `@autotvm.template` scope.

```{.python .input  n=2}
@autotvm.template
def auto_block_matmul(n):
    A, B, C = d2l.square_matmul(n)
    s = tvm.create_schedule(C.op)
    (x, y), (k,) = s[C].op.axis, s[C].op.reduce_axis

    cfg = autotvm.get_config()
    cfg.define_knob("tx", [1, 2, 4, 8, 16])
    cfg.define_knob("ty", [1, 2, 4, 8])
    cfg.define_knob("tk", [1, 2, 4, 8])

    xo, xi = s[C].split(x, cfg['tx'].val)
    yo, yi = s[C].split(y, cfg['ty'].val)

    ko, ki = s[C].split(k, cfg['tk'].val)

    s[C].reorder(xo, ko, yo, ki, xi, yi)
    s[C].vectorize(yi)
    s[C].unroll(xi)
    s[C].parallel(xo)

    return s, [A, B, C]
```

Then create a hyperparameter tuning task, we can print all searchable values by `config_space`.

```{.python .input  n=3}
n = 1024
task = autotvm.task.create(auto_block_matmul, args=(n,), target='llvm -mcpu=core-avx2')
print(task.config_space)
```

## Search the best hyperparameters 

The template we created has in total $5\times 5\times 8=200$ configurations. We can iterate all of them through exhaustive search. In general we often search in a large space that it's unlikely to try all possibilities. In machine learning, there are [multiple smart ways](https://en.wikipedia.org/wiki/Hyperparameter_optimization) to search in a large configuration space. The simple random search strategy, in each time it randomly samples a configuration, however, often works pretty well.

```{.python .input}
tuner = autotvm.tuner.RandomTuner(task)
```

The other two important considerations are the place to compile the program and the place to benchmark its performance. In a general case we can compile the program on a CPU server, and benchmark the performance on a arbitrary device, such as a mobile phone. In our case here both are on the local machine. But due to Jupyter's restriction on forking processes, we need to use the [RPC](https://en.wikipedia.org/wiki/Remote_procedure_call) protocol designed to for the general case. 

We first start the tracker. The tracker's job is dispatching workloads to various servers. We can run the following command by open a terminal in Jupyter (first click the Jupyter logo on top-right, next click "New" on top-left and select Terminal, then copy the following codes into the new Terminal window). 

`python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190`

Then we can open another terminal to start a server with a name `p3`, which connects to the tracker. 

`python -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=p3`

We can query the tracker status by:

```{.python .input}
!python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190
```

Once we see our server is listed, then create a program runner to connect to the tracker. Here we specify to repeat a workload by 10 times to get reliable results, and the connection timeout is 10 second.

```{.python .input}
runner = autotvm.RPCRunner(host='0.0.0.0', port=9190, key='p3',
                           number=10, timeout=10)
```

Now we can start the search. We randomly search 50 configurations and save the tuning results into a file. We can see that different hyperparameter configurations lead to dramatically different performances.

```{.python .input  n=4}
result_fname = 'block_matmul.log'
# The results are appended to the file, clear the previous results.
if os.path.exists(result_fname):
    os.remove(result_fname)
    
measure_option = autotvm.measure_option(builder='local', runner=runner)
tuner.tune(n_trial=50, measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file(result_fname)])
```

## Select the best hyperparameters

After the search is done, we can load the saved results to apply the best configuration.

```{.python .input  n=10}
import numpy as np

with autotvm.apply_history_best(result_fname):
    with tvm.target.create("llvm -mcpu=core-avx2"):
        s, args = auto_block_matmul(n)
        mod = tvm.build(s, args)
```

Then benchmark the performance.

```{.python .input}
# Because we used the magic function %timeit, it cannot be saved into the 
# d2ltvm package. So we copy it from the previous chapter.
def benchmark_square_matmul(func, n, constructor=None):
    x = np.random.normal(size=(n, n)).astype(np.float32)
    y = np.random.normal(size=(n, n)).astype(np.float32)
    z = np.empty_like(x)
    if constructor:
        x, y, z = constructor(x), constructor(y), constructor(z)
    res = %timeit -o -q -r3 func(x, y, z)
    return 2 * n ** 3 / res.average / 1e9

benchmark_square_matmul(mod, n, tvm.nd.array)
```

Compared to the result in :numref:`ch_matmul_cpu`, we can see that the searched hyperparameter performs better.

## Summary

- The performance relies on hyperparameters such as tiling sizes. The optimal value depends on the input sizes and hardware.
- We can specify a list of candidates for each hyperparameter and then search the optimal value

## Exercises 

- Change the hyperparameter candidates
- Apply the best searched hyperparameters into :numref:`ch_matmul_cpu`
