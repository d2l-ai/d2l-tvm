# Installation
:label:`ch_install`


To get you up and running with hands-on experiences, we'll need you to set up with a Python environment, Jupyter's interactive notebooks, the relevant libraries, and the code needed to *run the book*.

## Obtaining Source Codes

The source code package containing all notebooks is available at
http://tvm.d2l.ai.s3-website-us-west-2.amazonaws.com/d2l-tvm.zip.
Please download it and extract it into a
folder. For example, on Linux/macos, if you have both `wget` and `unzip`
installed, you can do it through:

```
wget http://tvm.d2l.ai.s3-website-us-west-2.amazonaws.com/d2l-tvm.zip
unzip d2l-tvm.zip -d d2l-tvm
```

## Installing Running Environment

If you have both Python 3.5 or later and pip installed, the easiest way to
install the running environment is through pip. There packages are needed,
`d2ltvm` for all dependencies such as Jupyter and saved code blocks, and `tvm`
for the deep learning compiler we are using. Some chapters use `mxnet` as
a baseline.

First install `d2ltvm`:

```
pip install git+https://github.com/d2l-ai/d2l-tvm
```

Then compile `tvm` from source codes. TVM doesn't have a pip package because it
highly depends on the libraries available on your system. Please follow the
instructions  on
[tvm.ai](https://docs.tvm.ai/install/from_source.html) to install tvm. The configration in `config.cmake` this
book requires includes

```
set(USE_CUDA ON)
set(USE_OPENCL ON)
set(USE_LLVM ON)
```

You can disable CUDA and OPENCL if no GPU is available on your machine. Also
don't forget the enable `cython`, which accelerates the performance. You just
need to run `make cython` in the TVM source folder.

If luckly you are using Ubuntu with `python-3.7`, `llvm-6.0` and `cuda-10.1` installed, you
may use the pre-built library that is for evaluating this book:

```bash
pip install https://tvm-repo.s3-us-west-2.amazonaws.com/tvm-0.6.dev0-cp37-cp37m-linux_x86_64.whl
```

Finally, install MXNet's CUDA version if GPUs are available. Assume you are have
CUDA 10.1 installed, then

```bash
pip install mxnet-cu101
```

You can change the `101` to match your CUDA version.

Once all packages are installed, we now open the Jupyter notebook by

```
jupyter notebook
```

At this point open http://localhost:8888 (which usually opens automatically) in the browser, then you can view and run the code in each section of the book.


### Code
:label:`ch_code`

We save reusable code blocks in the `d2ltvm` package by adding the comment `# Save to the
d2ltvm package.` before the code block. For example, the following is the
libraries imported by `d2ltvm`.

```{.python .input}
# Save to the d2ltvm package.
import tvm
import numpy as np
from matplotlib import pyplot as plt
from IPython import display
```
