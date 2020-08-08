# Art-Generation-With-Neural-Style-Transfer
Novel artistic images will be generated. Most of the algorithms optimize a cost function to get a set of parameter values. In Neural Style Transfer, we'll optimize a cost function to get pixel values!

This algorithm was created by [Gatys et al. (2015)](https://arxiv.org/abs/1508.06576).
# Problem Statement
Neural Style Transfer (NST) is one of the most fun techniques in deep learning. As seen below, it merges two images, namely: a "content" image (C) and a "style" image (S), to create a "generated" image (G).


The generated image G combines the "content" of the image C with the "style" of image S.


<p align = 'center'>
  <img src = '/images/louvre_generated.png'>
</p>

# Transfer Learning
Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that. The idea of using a network trained on a different task and applying it to a new task is called transfer learning.


Following the [original NST paper](https://arxiv.org/abs/1508.06576), we will use the VGG network. Specifically, we'll use VGG-19, a 19-layer version of the VGG network. This model has already been trained on the very large ImageNet database, and thus has learned to recognize a variety of low level features (at the shallower layers) and high level features (at the deeper layers).

# Neural Style Transfer (NST)
We will build the Neural Style Transfer (NST) algorithm in three steps:
- Build the content cost function  J<sub>content</sub>(C,G).
- Build the style cost function  J<sub>style</sub>(S,G).
- Put it together to get  J(G)=αJ<sub>content</sub>(C,G)+βJ<sub>style</sub>(S,G).

# Computing the content cost

##  Make generated image G match the content of image C
## Shallower versus deeper layers
- The shallower layers of a ConvNet tend to detect lower-level features such as edges and simple textures.
- The deeper layers tend to detect higher-level features such as more complex textures as well as object classes.

## Choose a "middle" activation layer  a[l] 
We would like the "generated" image G to have similar content as the input image C. Suppose you have chosen some layer's activations to represent the content of an image.

- In practice, you'll get the most visually pleasing results if you choose a layer in the middle of the network--neither too shallow nor too deep.

## Forward propagate image "C"
- Set the image C as the input to the pretrained VGG network, and run forward propagation.
- Let  a<sup>(C)</sup>  be the hidden layer activations in the layer you had chosen. This will be an  n<sub>H</sub>×n<sub>W</sub>×n<sub>C</sub>  tensor.

## Forward propagate image "G"
- Repeat this process with the image G: Set G as the input, and run forward progation.
- Let  a<sup>(G)</sup>  be the corresponding hidden layer activation.

# Content Cost Function  J<sub>content</sub>(C,G)
We will define the content cost function as:

<p align = 'center'>
  <img src = 'Screenshot(128).png'>
</p>
