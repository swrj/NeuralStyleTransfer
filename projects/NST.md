---
layout: post
title: 'Neural Style Transfer'
---
Our project implements Leon Gatys’ paper [*A Neural Algorithm of Artistic Style*](https://arxiv.org/pdf/1508.06576v2.pdf) which gave birth to a technique called Neural Style Transfer.  
[Github repo here](https://github.com/swrj/NeuralStyleTransfer)  

{% include image.html url="http://www.github.com/swrj/NeuralStyleTransfer" image="projects/NST/gene-kogan-nst.gif" %}
Pablo Picasso painting on glass in 1937, restyled by works from his Blue, African, and Cubist periods respectively. Image via [Gene Kogan](http://genekogan.com/works/style-transfer/)

## So what is Neural Style Transfer and why is it important?  
It is the formation of a new image by combining the style of an input image with the content of another input image. This outputs an image which retains the content of the original content input image, but transfers the style of the original input style image to the original content input image.  
These output images look stunning. If one uses a painting as the input style image, the resulting output looks like a piece of artwork painted by the artist.  The French have a word for this imitation called Pastiche, which roughly translates to “an artistic work in a style that imitates that of another work, artist, or period.
Humans are unique in their skill to create images that have a complex exchange between the content and style of an image and talented painters can easily create new pieces of art that imitate the style of a famous painter. It is however harder for a computer to create a pastiche. 
Neural Style Transfer is thus an important tool, not only because it can generate new images in the style of other images or paintings, but also because:  
* It shows how Neural Networks can be very powerful when used as a generative tool as opposed to a classification tool  
* It helps us get a deeper understanding on how Neural networks encode information that it deems important  

The first point is especially important as Neural Networks have long been used to classify or match patterns. Automatically generating content based on certain specifications is however still in its infancy. Making the most out of Neural Networks' generative capabilities could potentially help in various fields such as medicine, where one could theoretically design new molecules or proteins.

Neural Style Transfer has been used previously in commercial applications like Prisma and Ostagram which let users transform ordinary images into paintings. 
Neural Style Transfer has also been used in the production of animated movies as it reduces the number of frames that need to be painted or drawn, as was the case in the production of the short film Come Swim in 2017.  

There are many algorithms that attempt to implement Neural Style Transfer. The end result of Neural Style Transfer is art, and the quality of art is subjective. Although there are new algorithms that improve on some parts of Gatys' paper, they simultaneously compensate on other parts that are equally as important. 

{% include image.html url="http://www.github.com/swrj/NeuralStyleTransfer" image="projects/NST/conv.gif" %}

## How does it work?
A one line summary of Neural Style Transfer would be that it is basically an optimisation technique that uses a Convolutional Neural Network (CNN) to iteratively build the final image based by minimizing certain loss functions.  

To understand the algorithm completely, one needs to know how CNN's work. They are based on the principle of convolution, where a filter slides over every pixel in the image and produces an output which is essentially a transformation of the weighted sum of inputs covered by the filter. This output is called a feature map. Thus to summarize, the input image is convoluted with several filters, each with its own weights, to generate feature maps. These feature maps are then convoluted with more filters to create even more filter maps. Each of these layers provides a different interpretation of the input image.  
  
The maps in the lower layers of the CNN represent low level features such as lines or blobs while higher layers map out more complex features. A CNN thus represents a hierarchical representation of features of the input image.  
  
We thus need to somehow extract the "content" and the "style" of the input images by somehow manipulating the pixels in these feature layers.

We can extract the content and style of the input images by reducing the content and style cost functions over the pixels in the input images.
Content loss works on the intuition that images with similar content have similar representation in the higher layers of the network.  
On the other hand, in order to capture the style of the image, we need to capture the texture and pattern of the image. We thus mainly use the lower layers, which capture low level features very well.

Our content loss function is $${\sum(C^l(x) - C^l(h))^2}.$$ where $${C^l}.$$ is the CNN based layer function, $${l}.$$ is the layer of the network and $${x}.$$ is the current image (which we are manipulating to resemble the content of $${h}.$$, which is our input content image) 

Our style lost function is $${E(l) = (\frac{1}{\text{Normalization constants}})\sum(G^l - A^l)^2}.$$ where $${a}.$$ is the input image, $${x}.$$ is the image to be generated, $${G^l}.$$ and $${A^l}.$$ are gram matrices that are the inner product of feature maps in layer $${l}.$$ We use a Gram matrix in order to match feature distributions as the gram matrix contains non localized information about the image, such as the shapes, weights and textures in the image: essentially the style of the image. It is formed by multiplying the output matrix with it's transpose.

The total loss cost function is a representation of our core problem: minimizing the content loss and the style loss. We do this by changing the input to the VGG-19 network itself. We start off with a initial content image and slowly alter pixel values in order to minimize the content loss and style loss functions, which in turn minimizes the total loss function. We used the Adam optimizer to update network weights instead of a simple gradient descent optimizer to generate better results.

{% include image.html url="http://www.github.com/swrj/NeuralStyleTransfer" image="projects/NST/vgg19.jpg" %}
The VGG-19 Convolutional Neural Network  

## Our implementation
The VGG-19 Convolutional Neural Network is used in our implementation of Neural Style Transfer. VGG-19 won the ImageNet classification contest a couple of years back, so it is very adept at generating a rich hierarchical representation of features. This model was imported from Keras.
We made use of TensorFlow for carrying out most of the computation required for optimizing our loss functions. We used Numpy to do matrix arithmetic and image manipulations. 
The Gooey library was used for building a GUI.

Before running the algorithm we preprocess the input images by scaling them by a max dimension and subtract VGG_mean (used on original model for preprocessing) from the image. This process highlights the required features and reduces unwanted noise. This helps in taking different inputs and makes them uniform for the model to access and reduces computation.  

Similaraly we deprocess the output image before displaying and saving it.

## Additions to code

We added options so that the user has complete control over the output image. The user can choose the number of iterations to run (the higher the number, the more abstract the output and the longer it takes to run), the specific optimizer to use (we added support for Adam and Adagrad optimizers instead of a linear gradient descent optimizer as we saw better results). We also added options to allow the user to control the learning rate of the optimizer. In the case of the Adam optimizer, we also added the options to enter in custom values of beta1 (exponential decay rate for 1st moment estimates), beta2 (exponential decay rate for 2nd moment estimates) and epsilon (numerical stability constant) which in turn changes the output image to the users liking.

We have also made a simple GUI that provides an easier way to use the model. One can either run the model the easy way by just filling out the required arguments or one can fine tune the image by specifying the optional arguments. The GUI offers a streamlined simple way of running the algorithm without using the command line. We have packaged the GUI using pyinstaller so that the user can simply download it as a .exe file and run the algorithm without taking the time to set up the environment first.

{% include image.html url="http://www.github.com/swrj/NeuralStyleTransfer" image="projects/NST/GUI.jpg" %}

Finally we added the options of allowing the user to enter up to 3 style images to the style transfer algorithm. The algorithm transfer elements of style from each of these input images equally into the output.

## Results

If we use the Abe statue on Bascom Hill as our input content image and Van Gogh's self portrait as our input style image
{% include image.html url="http://www.github.com/swrj/NeuralStyleTransfer" image="projects/NST/Abe.jpg" %}
{% include image.html url="http://www.github.com/swrj/NeuralStyleTransfer" image="projects/NST/vangogh.jpg" %}
as shown above, we get the following sequence of images every 100 iterations
{% include image.html url="http://www.github.com/swrj/NeuralStyleTransfer" image="projects/NST/progression.jpg" %}
which eventually culminates in the following picture at a 1000 iterations of our algorithm
{% include image.html url="http://www.github.com/swrj/NeuralStyleTransfer" image="projects/NST/thumb.jpg" %}  