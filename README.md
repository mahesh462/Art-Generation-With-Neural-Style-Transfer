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

## Content Cost Function  J<sub>content</sub>(C,G)
We will define the content cost function as:

<p align = 'center'>
  <img src = 'Screenshot (128).png'>
</p>

- Here,  n<sub>H</sub>,n<sub>W</sub> and n<sub>C</sub> are the height, width and number of channels of the hidden layer you have chosen, and appear in a normalization term in the cost.
- For clarity, note that a<sup>(C)</sup> and a<sup>(G)</sup> are the 3D volumes corresponding to a hidden layer's activations.
- In order to compute the cost  J<sub>content</sub>(C,G) , it might also be convenient to unroll these 3D volumes into a 2D matrix, as shown below.
- Technically this unrolling step isn't needed to compute J<sub>content</sub>, but it will be good practice for when you do need to carry out a similar operation later for computing the style cost J<sub>style</sub>.
<p align = 'center'>
  <img src = '/images/NST_LOSS.png'>
</p>

# Computing the style cost

## Style Matrix

## Gram Matrix
- The style matrix is also called a "Gram matrix."
- In linear algebra, the Gram matrix G of a set of vectors  (v1,…,vn)  is the matrix of dot products, whose entries are  G<sub>ij</sub>=v<sup>T</sup><sub>i</sub>v<sub>j</sub>.
- In other words, G<sub>ij</sub> compares how similar v<sub>i</sub> is to v<sub>j</sub>: If they are highly similar, you would expect them to have a large dot product, and thus for G<sub>ij</sub> to be large.
## Two meanings of the variable  G
- Note that there is an unfortunate collision in the variable names used here. We are following common terminology used in the literature.
- G  is used to denote the Style matrix (or Gram matrix)
- G  also denotes the generated image.
- To avoid confusion, we will use  G<sub>gram</sub> to refer to the Gram matrix, and  G to denote the generated image.
## Compute  G<sub>gram</sub>
In Neural Style Transfer (NST), you can compute the Style matrix by multiplying the "unrolled" filter matrix with its transpose:
<p align = 'center'>
  <img src = '/images/NST_GM.png'>
</p>

## G<sub>(gram)i,j</sub>: correlation
The result is a matrix of dimension  (n<sub>C</sub>,n<sub>C</sub>)  where  n<sub>C</sub>  is the number of filters(channels). The value  G<sub>(gram)i,j</sub>  measures how similar the activations of filter i are to the activations of filter j.
## G<sub>(gram)i,i</sub>: prevalence of patterns or textures
- The diagonal elements G<sub>(gram)i,i</sub> measure how "active" a filter i is.
- For example, suppose filter  i  is detecting vertical textures in the image. Then G<sub>(gram)i,i</sub> measures how common vertical textures are in the image as a whole.
- If G<sub>(gram)i,i</sub> is large, this means that the image has a lot of vertical texture.

By capturing the prevalence of different types of features (G<sub>(gram)i,i</sub>), as well as how much different features occur together (G<sub>(gram)i,j</sub>), the Style matrix  G<sub>gram</sub>  measures the style of an image.

## Style cost
Our goal will be to minimize the distance between the Gram matrix of the "style" image S and the gram matrix of the "generated" image G.
- For now, we are using only a single hidden layer a<sup>[l]</sup>.
- The corresponding style cost for this layer is defined as:
<p align = 'center'>
  <img src = 'Screenshot (130).png'>
</p>

- G<sup>S</sup><sub>gram</sub> Gram matrix of the "style" image.
- G<sup>G</sup><sub>gram</sub> Gram matrix of the "generated" image.
- This cost is computed using the hidden layer activations for a particular hidden layer in the network  a<sup>[l]</sup>.

# Style Weights
- So far you have captured the style from only one layer.
- We'll get better results if we "merge" style costs from several different layers.
- Each layer will be given weights(λ<sup>[l]</sup>) that reflect how much each layer will contribute to the style.
- By default, we'll give each layer equal weight, and the weights add up to 1. (∑<sub>l</sub><sup>L</sup> λ<sup>[l]</sup> = 1).

Combine the style costs for different layers as follows:
<p align = 'center'>
  <img src = 'Screenshot (131).png'>
</p>

# Total cost
<p align = 'center'>
  <img src = 'Screenshot (132).png'>
</p>
