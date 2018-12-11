---
layout: post
title: 'Neural Style Transfer'
---
# Artistic Style Transfer

###### [Original Project Proposal](https://docs.google.com/document/d/1SkfLmti0sP-YI5Gd3YDAnLqvERRPuLpEeC7HIRwHeAY/edit?usp=sharing)

## Timetable
* Mid October: Figure out whether an interactive web app is feasible, if not we will fallback to an app with a GUI.
* October 31: [Midterm Report](swrj.github.io/NeuralStyleTransfer/Report.html)
* Mid November: Finish the model, at this point it should work and generate images
* Start of December: Finish the GUI, if all goes well we can try supporting video files as well
* December 3: Final project presentation
* December 12: Project webpage

## Proposal
Our project will implement the paper *A Neural Algorithm of Artistic Style* by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge along with a couple of additions. We found this paper after reading up about style transfer after seeing Gene Kogan’s synthesized images. The paper proposes a novel way to generate artistic images that combines the style of an existing image with the content of another existing image to create a new merged image. 

The paper shows that the task of transferring style from one image to another can be thought of simply as an optimization problem that can be solved through training a convolutional neural network. A convolutional neural network that has been trained to detect objects has its own internal and independent representations of content and style within an image. We can then use a pre trained convolutional neural network to extract a style representation from an image and apply it to a content representation of another image to create an entirely new image that retains the content of the second image but with the style of the first. When the paper was written in 2015, the VGG convolutional network was the best at identifying images. We plan on using a pre trained Squeeze and Excitation Neural Network (which won the ImageNet 2017 Classification Task) in place of the VGG network to improve the implementation of the paper.

We plan on creating a simple web app with a GUI that allows users to choose pre-selected sample content and sample images to merge. The GUI will also have plenty of options and parameters that allow the user to have fine grain control over the output image such as number of iterations, style blend weights (when there are multiple input style images), style layer weights (to make the image look more “abstract” or more “concrete”), content weights (to choose a finer or coarser grain content transfer between pictures), pooling types (eg: max vs average pooling), etc. We plan on coding the model in Python (using TensorFlow), writing the webpage in Javascript (or alternatively using a github.io page) and hosting the project on AWS.

##Team Members:
**Swaraj Rao**, srao24@wisc.edu, srao24
**Sai Rohit Battula**, battula2@wisc.edu, battula2
