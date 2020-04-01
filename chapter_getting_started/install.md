# Installation
:label:`ch_install`

Each section of this book is a Jupyter notebook. The easiest way to run them is clicking the **COLAB** button on the upper right of the HTML page, which will directly you to Google Colab with the corresponding notebook opened. Running the first code cell will connect to a host runtime and show the following warning message. You can click RUN ANYWAY to continue.

![Click RUN ANYWAY to run a section on Colab.](../img/colab.png)
:width:`400px`
:label:`img_colab`

The reset of this section will go through how to set up a Python environment, Jupyter's interactive notebooks, the relevant libraries, and the code needed to run the book you can run them on your machines.

## Obtaining Source Codes

The source code package containing all notebooks is available at
http://tvm.d2l.ai/d2l-tvm.zip.
Please download it and extract it into a
folder. For example, on Linux/macOS, if you have both `wget` and `unzip`
installed, you can do it through:

```bash
wget http://tvm.d2l.ai/d2l-tvm.zip
unzip d2l-tvm.zip -d d2l-tvm
```


## Installing Running Environment

If you have both `Python 3.5` or later and `pip` installed, the easiest way to
install the running environment is through `pip`. The required packages are

- `d2ltvm` for all dependencies such as Jupyter and saved code blocks
- `tvm` :cite:`Chen.Moreau.Jiang.ea.2018` for the deep learning compiler we are using
- `mxnet` as the baseline in some chapters

First install `d2ltvm`:

```bash
pip install git+https://github.com/d2l-ai/d2l-tvm
```


Then compile `tvm` from source codes. TVM doesn't have a pip package because it
highly depends on the libraries available on your system. Please follow the
instructions  on
[tvm.ai](https://docs.tvm.ai/install/from_source.html) to install `tvm`. The configuration in `config.cmake` this
book requires at least

```bash
set(USE_LLVM ON)
```


If you plan to run on Nvidia GPUs as well, you will also need to

```bash
set(USE_CUDA ON)
```

Also
don't forget to enable `cython`, which accelerates the performance. You just
need to run `make cython` in the TVM source folder.

If luckily you are using Ubuntu with `python-3.7`, `llvm-6.0` and `cuda-10.1` installed, you
may use the pre-built library that is for evaluating this book:

```bash
pip install https://tvm-repo.s3-us-west-2.amazonaws.com/tvm-0.7.dev1-cp37-cp37m-linux_x86_64.whl
```

Our code runs on tvm-0.7-dev1 for now.

Finally, install MXNet's CUDA version if GPUs are available :cite:`Chen.Li.Li.ea.2015`. Assume you are have
CUDA 10.1 installed, then

```bash
pip install mxnet-cu101
```


You can change the `101` to match your CUDA version.

Once all packages are installed, you can open the Jupyter notebook by

```bash
jupyter notebook
```


At this point open http://localhost:8888 (which usually opens automatically) in the browser, then you can view and run the code in each section of the book.


## Code
:label:`ch_code`

Throughout the book, we save reusable code blocks in the `d2ltvm` package by adding the comment: "`# Save to the
d2ltvm package.`" before the code block. For example, the following code snippet shows the
libraries imported by `d2ltvm`.

```{.python .input  n=1}
# Save to the d2ltvm package.
import tvm
from tvm import te
import time
import timeit
import numpy as np
from matplotlib import pyplot as plt
from IPython import display
try:
  import mxnet as mx
except:
  pass
```

## [Discussions](https://discuss.tvm.ai/t/getting-started-installation/4706)
