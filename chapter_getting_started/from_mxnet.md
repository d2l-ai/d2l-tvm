# Neural Network Inference
:label:`ch_from_mxnet`

You have seen how to implement and compile a simple vector addition operator in :numref:`ch_vector_add`. Now we will make a big jump to compile a whole pre-trained neural network, which consists of a set of operators, to run the inference.

```{.python .input  n=1}
import numpy as np
import mxnet as mx
from PIL import Image
import tvm 
from tvm import relay
```

Here three additional modules are imported than the previous chapter. We will use `PIL` to read images, `MXNet` to obtain pre-trained neural networks, and the `relay` module in TVM to convert and optimize a neural network. 

## Obtaining Pre-trained Models

A pre-trained model means a neural network with parameters trained on a data set. Here we download and load a ResNet-18 model by specifying `pretrained=True` from MXNet's model zoo. If you want to know details about this model, please refer to [Chapter 7.6 in D2L](http://d2l.ai/chapter_convolutional-modern/resnet.html). You can find more models on the [MXNet model zoo](https://mxnet.apache.org/api/python/docs/api/gluon/model_zoo/index.html) page, or refer to [GluonCV](https://gluon-cv.mxnet.io/model_zoo/index.html) and [GluonNLP](http://gluon-nlp.mxnet.io/model_zoo/index.html) for more computer vision and natural language models.

```{.python .input  n=2}
model = mx.gluon.model_zoo.vision.resnet18_v2(pretrained=True)
len(model.features), model.output
```

The loaded model is trained on the Imagenet 1K dataset, which contains around 1 million natural object images among 1000 classes. The model has two parts, the main body part `model.features` contains 13 layers, and the output layer is a dense layer with 1000 outputs. 

The following code block loads the text labels for each class in the Imagenet dataset.

```{.python .input  n=3}
with open('../data/imagenet1k_labels.txt') as f:
    labels = eval(f.read())
```

## Pre-processing Data

We first read a sample image. It is resized to the size, i.e. 224 px width and height, we used to train the neural network.

```{.python .input  n=4}
image = Image.open('../data/cat.jpg').resize((224, 224))
image
```

According to the [model zoo page](https://mxnet.apache.org/api/python/docs/api/gluon/model_zoo/index.html). Image pixes are normalized on each color channel, and the data layout is `(batch, RGB channels, height and width)`. The following function transforms the input image to satisfies the requirement.

```{.python .input  n=5}
# Save to the d2ltvm package
def image_preprocessing(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image.astype('float32')

x = image_preprocessing(image)
x.shape
```

## Compile Pre-trained Models

To compile a model, we first convert the MXNet model into relay. In the `from_mxnet` function, we provide the model with the the input data shape. We could specify an undetermined shape, which is necessary for some neural networks. But a fixed shape often leads to a compact library and a better performance.

```{.python .input  n=6}
relay_mod, relay_params = relay.frontend.from_mxnet(model, {'data': x.shape})
type(relay_mod), type(relay_params)
```

This function will return the program `relay_mod`, which is a relay module, and a dictionary of parameters `relay_params` that maps a string key to a TVM ndarray. Next, we compile them using the `llvm` backend, which is recommended for CPUs. [LLVM](https://en.wikipedia.org/wiki/LLVM) defines an intermediate representation that has been adopted by multiple programming languages. The LLVM compiler is then be able to compile the generated programs into machine codes. TVM generate LLVM codes for CPUs, and then invoke LLVM to compile to the machine codes. We have already used it to compile the vector addition operator in the last chapter, despite that we didn't explicitly specify it.  

In addition, we set the optimization level to the highest level 3. You may get warning messages that not every operator is well optimized, you can ignore it for now. We will get back to it later.

```{.python .input  n=7}
target = 'llvm'
with relay.build_config(opt_level=3):
    graph, mod, params = relay.build(relay_mod, target, params=relay_params)
```

The compiled module has three parts: `graph` is a json string described the neural network, `mod` contains all compiled operators used to run the inference, and `params` is dictionary mapping parameter name to weights.

```{.python .input  n=8}
type(graph), type(mod), type(params)
```

You can see `mod` is a TVM module we already seen in :numref:`ch_vector_add`. 

## Inference

Now we can create a runtime to run the inference, namely the forward pass. Creating the runtime needs the program and library, with a device context that can be constructed from the target. The device is the CPU here. Next we specify the parameters with `set_input` and run the workload by giving the input data. Since this network has a single output layer, we can obtain it, a `(1, 1000)` shape matrix, by `get_output(0)`. The final output is a 1000-length NumPy vector.

```{.python .input  n=9}
ctx = tvm.context(target)
rt = tvm.contrib.graph_runtime.create(graph, mod, ctx)
rt.set_input(**params)
rt.run(data=tvm.nd.array(x))
scores = rt.get_output(0).asnumpy()[0]
scores.shape
```

The vector contains the predicted confidence score for each class. Note that the pre-trained model doesn't have the [softmax](https://en.wikipedia.org/wiki/Softmax_function) operator, so these scores are not mapped into probabilities in (0, 1). Now we can find the two largest scores and report their labels.

```{.python .input  n=10}
a = np.argsort(scores)[-1:-5:-1]
labels[a[0]], labels[a[1]]
```

## Saving the Compiled Library

We can save the output of `relay.build` in disk to reuse them later. The following codes block saves the json string, library, and parameters.

```{.python .input  n=11}
!rm -rf resnet18*

name = 'resnet18'
graph_fn, mod_fn, params_fn = [name+ext for ext in ('.json','.tar','.params')]
mod.export_library(mod_fn)
with open(graph_fn, 'w') as f:
    f.write(graph)
with open(params_fn, 'wb') as f:
    f.write(relay.save_param_dict(params))

!ls -alht resnet18*
```

Now we load the saved module back.

```{.python .input  n=12}
loaded_graph = open(graph_fn).read()
loaded_mod = tvm.module.load(mod_fn)
loaded_params = open(params_fn, "rb").read()
```

And then construct the runtime as before to verify the results

```{.python .input  n=13}
loaded_rt = tvm.contrib.graph_runtime.create(loaded_graph, loaded_mod, ctx)
loaded_rt.load_params(loaded_params)
loaded_rt.run(data=tvm.nd.array(x))
loaded_scores = loaded_rt.get_output(0).asnumpy()[0]
tvm.testing.assert_allclose(loaded_scores, scores)
```

## Summary

- We can use the relay module to convert and compile a neural network.
- We can save the compiled module into disk to facilitate future deployment.