---
layout: post
title: 'Project Midterm Report (10/31/18)'
---

# Midterm Report

###### [Original Project Midterm Report](https://docs.google.com/document/d/1mSnwTLdyXf_L5wPUK20Ozkspxf6ZbtTApwF5ZWDXgzE/edit?usp=sharing)

## Explain problem
We decided to implement A Neural Algorithm of Artistic Style by Gatys et. al. This paper provides a method for extracting the “style” from one image and the “content” from another image and forming a new image with the extracted content and style. Previously, it was not known if the representations of content and style of images were truly separable using Convolutional Neural Networks. 

The paper treats the problem of isolating a picture’s content and style as an optimisation problem. A white noise image is the starting point, and is made more and more similar to the desired merged image through the optimization of a cost function. This was a novel way of tackling the problem of style transfer as instead of updating the weights and biases (as in a traditional CNN), they are kept constant and the pixels are modified instead. 

##Why is it important
Neural style transfer can generate images in the styles of famous painters. In the past, creating images in the style of another painter required a highly skilled artist and a lot of time. Many commercial applications such as the mobile application Prisma and the web application Ostagram have monetized the appeal of neural style transfer as a simple way of allowing users to create art. There are several potential applications for this technology. For example, neural style transfer could be used in the production of animated movies. Creating an animation requires about 24 frames per second, which are usually painted or drawn. Neural style transfer could be used to automatically stylise the frames into a specific animation style quickly, as was the case in the production of the short film Come Swim in 2017.

##Current state of the art
Gatys et al.’s work in 2015 led to a series of advances in Neural Style algorithms. These algorithms can be broadly divided into Image Optimisation Based Online Neural Methods (IOB-NST) and Model Optimisation Based Offline Neural Methods (MOB-NST). Gatys et al.’s method does not perform well in preserving the fine structure and details of the input content image. This method also does not take into account the different depth information and low level information in the input content image. 

IOB-NST algorithms (such as Gatys et al.’s algorithm) first model and extract style and content information, recombine them as the target representation and then iteratively reconstruct a stylized image that closely matches the target representation. These algorithms are very computationally expensive due to the iterative image optimisation procedure.

Risser et al found that the usage of a Gram matrix introduces instabilities during optimisation due to the fact that feature activations with different means and variances can still have the same Gram matrix. Risser et al. introduces the concept of extra histogram loss which forces the optimisations to match the entire histogram of feature activations. This results in a more stable Neural Style transfer with fewer parameter tunings and iterations. This however fails to fix the other weaknesses of Gatys et al.’s work: namely the lack of consideration towards the depth and low level information of the input content image. 

Li et al introduces an additional Laplacian loss (which is usually used for edge detection) to incorporate constraints upon low level features in the pixel space. This algorithm shows better performance is preserving finer details and structures but fails to improve on Gatys et al.’s algorithm when it comes to considering semantics and depth variations.

Li and Wand used Markov Random Fields to preserve fine structure and arrangement better. This algorithm is great for photorealistic styles but fails when the content and style images have strong differences.
MOB-NST uses Model Optimisation Based Offline Image Reconstruction to reconstruct the stylised result by optimising a feed forward network over a large set of style images.

Ulyanov et al. use a per style per model by pre training a feed forward style specific network to produce a stylised result with a single forward pass (which results in real time transfer). Ulyanov et al. use normalization to every single image to improve stylisation quality.

There are thus many algorithms that try to improve Neural Style Transfer. Different algorithms produce different results (different stroke sizes, different levels of abstractness, etc.). However, the quality of the output image is ultimately subjective and one should choose an algorithm that pertains to ones intended use case.

## Reimplementing existing solution or proposing new approach?
We are planning on re-implementing an existing solution proposed by Gatys et al. with some changes and additions. We are planning on implementing the solution in the following way:

* We will be making use of a pre-trained image classification CNN based model (VGG 19 was used in the paper) as the intermediate layers in order to provide essential feature maps which would then be used for generating content and style properties of the input images. We will implement the algorithm using the VGG 19 image classifier first, however we would like to check how the algorithm would perform against other successful image classifiers such as Inception or ResNet.
The goal of the model is to minimize the cost functions associated with content and style. 
* For any image ‘x’ and input content image ‘h’ , if we represent the CNN based layer function as , the content loss function  =   (l represents the layer of the network). We are going to make use of backpropagation to minimize the error function.
* Let ‘a’ be the input image, ‘x’ be an arbitrary image (image to be generated),  and  be respective gram matrices (inner product of feature maps in layer l) then the cost function for style is given by . We minimize the cost function making use of gradient descent.
* We will iteratively compute losses and record the appropriate loss and the corresponding output image.
* We plan on implementing this algorithm using either the Adam or L-BFGS Tensorflow optimizers.
* We plan on extending this algorithm to videos.


## How will you evaluate performance of your solution? Results and comparisons that you will show
The output images are subjective and thus there is no definitive criterion for evaluating how good the output images are. However, we are planning on presenting a wide range of output images with varying parameters such as no. of iterations, content weights and style weights, for a given input image. These options would be built into our GUI so that the user can choose the output image subjectively.

If we manage to successfully produce an alternate model with a different image classifier as mentioned above, we could provide a comparison between the output images produced by both the models with the same parameters. We would further, present standardized benchmark comparisons such as memory footprint and time taken for the model to produce an acceptable/similar image to further evaluate both the models. However, what constitutes as an acceptable image produced by both the models is also subjective. 

## Current Progress
Our current progress involves the following: 
* As both of us have limited experience with machine learning (mostly limited to linear models) and image analysis, we had to learn core concepts involving learning models, convolutional neural networks, minimization techniques such as gradient descent and backpropagation, and about the data science pipeline involving pre-processing and visualization techniques.
* We implemented sample neural nets dealing with various activation functions such as ReLu and sigmoid, and made use of various minimization techniques mentioned above in tensorflow to help us effectively implement the above solution.
* We are currently exploring the use of pre-trained models in tensorflow and the use of intermediate layers for minimizing content and style loss functions.
We are currently testing out both Kivy and PyGUI in order to build our GUI. We have scrapped the idea of a web application.

## Future Timeline
Start of November: Choose between Kivy and PyGUI for our GUI library

Mid November: Finish writing most of the python code and start writing code for the GUI

End of November: See if its possible to extend the algorithm to videos and compare performance of VGG 19 classifier with Inception and ResNet

## References
https://arxiv.org/pdf/1508.06576v2.pdf
https://shafeentejani.github.io/2016-12-27/style-transfer/
https://arxiv.org/pdf/1705.04058.pdf
https://arxiv.org/pdf/1610.07629.pdf
https://ai.googleblog.com/2016/10/supercharging-style-transfer.html
http://openaccess.thecvf.com/content_cvpr_2017/papers/Gatys_Controlling_Perceptual_Factors_CVPR_2017_paper.pdf
https://medium.com/artists-and-machine-intelligence/neural-artistic-style-transfer-a-comprehensive-look-f54d8649c199
https://towardsdatascience.com/a-brief-introduction-to-neural-style-transfer-d05d0403901d
https://arxiv.org/pdf/1701.01036.pdf
https://arxiv.org/pdf/1701.04928.pdf
https://blog.slavv.com/picking-an-optimizer-for-style-transfer-86e7b8cba84b
