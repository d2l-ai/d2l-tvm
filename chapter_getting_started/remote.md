# Running on a Remote Machine
:label:`ch_remote`

In this book, we will run and optimize programs on various hardware platforms. One way is to log into the machine with the desired hardware, install required packages and then run the workloads there. It, however, makes maintaining the source codes and data difficult, especially when the targeting hardware is with minimal power. In this session, we will describe another solution: running a daemon on the remote machine and then sending the compiled module and input data to it only for execution.

```{.python .input  n=1}
import d2ltvm
import numpy as np
import mxnet as mx
import tvm
from tvm import rpc, relay
from PIL import Image
```

Note that we imported the `rpc` module from TVM. [RPC](https://en.wikipedia.org/wiki/Remote_procedure_call), namely remote procedure call, enables executing a program on a remote place.

## Setup the Remote Machine

We first need to install TVM `runtime` module on the remote machine. The installation setup is almost identical to TVM (refer to :numref:`ch_install`), except that we only need to build the runtime, i.e. `make runtime`, instead of the whole TVM library. The runtime size is often less than 1MB, which makes it suitable for device with memory constraints. You also need to enable the proper backend, e.g. `CUDA` or `OpenCL`, if necessary.

Once the runtime is installed, we can start the daemon by running the following command on the remote machine.

`python -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090`

It will start an RPC server which binds the local 9090 port to listen. You should see the following output indicating the server has already started.

`INFO:RPCServer:bind to 0.0.0.0:9090`

In addition, you need to check two things on the remote machine.

One is the remote machine's IP. On Linux or macOS, you can get it by `ifconfig  | grep inet`. Also remember to open the 9090 port if there is a firewall.

The other one is the target architecture. It's straightforward for GPUs, we will cover it later. For CPUs, the easiest way is installing LLVM on the remote machine and then checking `llvm-config --host-target`. The return of the remote machine we are using is `x86_64-pc-linux-gnu`.

This target triplet has the general format `<arch><sub>-<vendor>-<sys>-<abi>`, where

- arch: x86, x86_64, arm, thumb, mips, etc.
- sub: for ARM, there are v5, v6m, v7a, v7m, v8, etc.
- vendor: pc, apple, nvidia, ibm, etc.
- sys: linux, win32, darwin, cuda, none, unknown, etc.
- abi: eabi, gnu, android, macho, elf, etc.

For example, it's `x86_64-apple-darwin17.7.0` for the MacbookPro I'm using, and `armv6k-unknown-linux-gnueabihf` for the Raspberry Pi 4B.


## Compile the Program for the Remote Machine

Let's run the vector addition defined :numref:`ch_vector_add` on the remote machine. Note that we specified the remote machine target through the `-target` argument for LLVM.

```{.python .input  n=2}
n = 100
target = 'llvm -target=x86_64-pc-linux-gnu'

args = d2ltvm.vector_add(n)
s = tvm.create_schedule(args[-1].op)
mod = tvm.build(s, args, target)
```

Then we save the compiled module to disk, which will be uploaded to the remote machine later.

```{.python .input  n=3}
mod_fname = 'vector-add.tar'
mod.export_library(mod_fname)
```

## Evaluate on the Remote Machine

We first connect to the remote machine with the IP we checked before.

```{.python .input  n=4}
remote = rpc.connect('172.31.0.149', 9090)
```

Next, we send the compiled library to the machine and load it into the memory of the remote machine.

```{.python .input  n=5}
remote.upload(mod_fname)
remote_mod = remote.load_module(mod_fname)
```

When creating the data, we specify the device context as CPU on the remote machine. The data will be created on the local machine as before, but will be sent to the remote machine later. Note that we used NumPy to create the data, but there is no need to have the remote machine also installed NumPy.

```{.python .input  n=19}
ctx = remote.cpu()
a, b, c = d2ltvm.get_abc(n, lambda x: tvm.nd.array(x, ctx=ctx))
```

Since both data and library are ready on the remote machine, let's execute the program on it as well.

```{.python .input  n=26}
remote_mod(a, b, c)
```

Finally, the `.asnumpy()` method will send the data back to the local machine and convert to a NumPy array. So we can verify the results as before.

```{.python .input}
np.testing.assert_equal(a.asnumpy()+b.asnumpy(), c.asnumpy())
```

## Running Neural Network Inference

Let's run the ResNet-18 used in :numref:`ch_from_mxnet` on the remote machine. As before, we load a sample image and Imagenet 1K labels.

```{.python .input  n=9}
image = Image.open('../data/cat.jpg').resize((224, 224))
x = d2ltvm.image_preprocessing(image)
with open('../data/imagenet1k_labels.txt') as f:
    labels = eval(f.read())
```

Then we convert, compile and save the module. Note that we just need to save the shared library which contains the machine code of the compiled operators to the disk.

```{.python .input  n=10}
mod_fname = 'resnet18.tar'
model = mx.gluon.model_zoo.vision.resnet18_v2(pretrained=True)
relay_mod, relay_params = relay.frontend.from_mxnet(model, {'data': x.shape})
with relay.build_config(opt_level=3):
    graph, mod, params = relay.build(relay_mod, target, params=relay_params)
mod.export_library(mod_fname)

```

Next, we upload the saved library to the remote machine and load it into memory. Then we can create a runtime using the model definition, the remote library and the remote context.

```{.python .input  n=14}
remote.upload(mod_fname)
remote_mod = remote.load_module(mod_fname)
remote_rt =  tvm.contrib.graph_runtime.create(graph, remote_mod, ctx)

```

The inference is identical to :numref:`ch_from_mxnet`, where both parameters and input are on the local machine. The runtime will upload them into the remote machine properly.

```{.python .input  n=15}
remote_rt.set_input(**params)
remote_rt.run(data=tvm.nd.array(x))
scores = remote_rt.get_output(0).asnumpy()[0]
scores.shape
a = np.argsort(scores)[-1:-5:-1]
labels[a[0]], labels[a[1]]
```

## Summary

- We can install a TVM runtime on a remote machine to start an RPC server to accept workloads to run.
- A program can be compiled locally with specifying the remote machine's architecture target (called cross-compilation), and then run on the remote machine via RPC.

## [Discussions](https://discuss.tvm.ai/t/getting-started-running-on-a-remote-machine/4709)
