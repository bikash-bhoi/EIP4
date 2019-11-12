**Name : Bikash Ranjan Bhoi**

**Reg. email Id : bhoi.bikash@gmail.com**

**Batch M6**



### 1.Convolution

In Computer Vision, Convolution is used to extract important feature of a dataset using dot products two matrices ,the kernel and a portion of the input dataset.



### 2.Kernel/Filter

Filter/Kernel is a small matrix used to extract specific pattern of a given data-set. Good Practice is to use a 3x3 Matrix. 



### 3.Epochs

An Epoch in ANN is a full traversal of input data through the Network, which includes Both forward and back propagation. At the end of an Epoch, the model weights are updated  by backpropagaition fitting better on training data.



### 4.1x1 Convolution

1x1 Convolution uses filter of size 1x1, however  it has the capability to alter the number of channels from earlier layer. It is used to reduce the dimensionality for faster computation without losing much of relevant data.



### 5. 3x3 Convolution

3x3 Convolution uses filter of size 3x3 that moves over the input data-set and performs convolution to produce the output of the layer. It helps extract smaller feature of an Image.



### 6. Feature Maps

Feature Map is the output of a convolution layer. The Feature Maps are responsible for detecting Edges, Gradients, patterns, textures, part of object so they should be visualized to understand how a network is learning.



### 7. Activation Function

Activation functions define the output of a particular neuron of the Neural Network. They scale the output value to a specific range adding/not adding non-linearity or they can filter some value out. The common Activation function used are ReLu, Softmax, Logistic function, Leaky ReLu, Binary Step Function etc.

ReLU : (Image source: Kaggle)

![ReLU image](https://i.imgur.com/gKA4kA9.jpg)



### 8. Receptive Field

The receptive field of an unit in layer is the area that it is affected by from the previous layer. The Local receptive field is the area on which the convolution/maxpool happened, where as the Global receptive field is the whole area of input image from which the unit is calculated.