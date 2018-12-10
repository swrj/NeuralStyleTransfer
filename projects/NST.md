---
layout: post
title: 'Neural Style Transfer'
---
Our project implements Leon Gatys’ paper ‘A Neural Algorithm of Artistic Style’ which gave birth to Neural Style Transfer

![alt text](projects/NST/gene-kogan-nst.gif "Pablo Picasso painting on glass in 1937, restyled by works from his Blue, African, and Cubist periods respectively. Image via Gene Kogan")

So what is Neural Style Transfer?
It is the formation of a new image by combining the style of an input image with the content of another input image.
The French have a word for this called Pastiche which is roughly translated to “an artistic work in a style that imitates that of another work, artist, or period.”
Humans are unique in their skill to create images that have a complex exchange between the content and style of an image and talented painters can easily create new pieces of art that imitate the style of a famous painter. It is however harder for a computer to create a pastiche. 
We will thus be implementing Gatys’ paper with a few additions.

![alt text](projects/NST/conv.gif "Convolution")

Neural Style Transfer has been used previously in commercial applications like Prisma and Ostagram which let users transform ordinary images into paintings. 
Neural Style Transfer has also been used in the production of animated movies as it reduces the number of frames that need to be painted or drawn, as was the case in the production of the short film Come Swim in 2017.

Neural Style Transfer is basically an optimisation technique that uses a Convolutional Neural Network to iteratively build the final image.
The VGG-19 Convolutional Neural Network is used in our implementation of Neural Style Transfer. VGG-19 won the ImageNet classification contest a couple of years back, so it is very adept at generating a rich hierarchical representation of features. 
We are especially interested in the maps in the lower layers  and higher layers as the maps in the lower layers look for low level features such as lines or blobs while maps in the higher layers look for more complex features. 
Unlike updating weights and biases, as in a traditional problem involving a CNN, style transfer starts with a blank image. A cost function is then constantly optimized by changing the pixel values of the image.

{% include image.html url="http://www.github.com/swrj/NeuralStyleTransfer" image="projects/NST/vgg.jpg" %}
The VGG-19 Convolutional Neural Network

We made use of TensorFlow for carrying out most of the computation required for optimizing our loss functions. We used Numpy to do matrix arithmetic and image manipulations. 
The VGG-19 model and the corresponding pre trained intermediate layers were imported from keras while the Gooey library was used for building the GUI.

First we preprocess the input images by scaling them by a max dimension and subtract VGG_mean (used on original model for preprocessing) from the image. This process highlights the required features and reduces unwanted noise. This helps in taking different inputs and makes them uniform for the model to access and reduces computation.

We can extract the content and style of the input images by reducing the content and style cost functions over the pixels in the input images.
Content loss works on the intuition that images with similar content have similar representation in the higher layers of the network
In order to capture the style of the image, we need to capture the texture and pattern of the image. We thus mainly use the lower layers, which capture low level features very well. We use a Gram matrix in order to match feature distributions, the style loss is thus simply the normalized, squared difference in Gram matrices between the two images.

Our content loss function is $${\sum(C^l(x) - C^l(h))^2}.$$ where $${C^l}.$$ is the CNN based layer function, $${l}.$$ is the layer of the network and $${x}.$$ is the current image (which we are manipulating to resemble the content of $${h}.$$, which is our input content image) 

Our style lost function is $${E(l) = (\frac{1}{\text{Normalization constants}})\sum(G^l - A^l)^2}.$$ where $${a}.$$ is the input image, $${x}.$$ is the image to be generated, $${G^l}.$$ and $${A^l}.$$ are gram matrices that are the inner product of feature maps in layer $${l}.$$

The total loss cost function is a representation of our core problem: minimizing the content loss and the style loss. We do this by changing the input to the VGG-19 network itself. We start off with a initial content image and slowly alter pixel values in order to minimize loss. We used the Adam optimizer to update network weights instead of a simple gradient descent optimizer to generate better results.

We have also made a simple GUI that provides an easier way to use the model. One can either run the model the easy way by just filling out the required arguments or one can fine tune the image by specifying the optional arguments. These arguments range from the number of iterations to run for to the specific content and style weights which can end up changing the final image drastically.

{% include image.html url="http://www.github.com/NeuralStyleTransfer" image="projects/NST/GUI.jpg" %}
