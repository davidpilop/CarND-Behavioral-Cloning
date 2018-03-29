# Udacity Self-Driving Car Engineer Nanodegree - Behavioral Cloning Project

*My solution to the Udacity Self-Driving Car Engineer Nanodegree Behavioral Cloning project.*

**Note: This project makes use of a Udacity-developed driving simulator and training data collected from the simulator (neither of which is included in this repo).**

## Introduction

The challenge of this project is not only developing a CNN model that is able to drive the car around the test track without leaving the track boundary, but also feed data to the CNN in a way that allows the model to generalize well enough to drive in an environment it has not yet encountered (i.e. the challenge track).

In this project, we use deep learning to imitate human driving in a simulator. In particular, we utilize Keras libraries to build a convolutional neural network that predicts steering angle response in the simulator.

### Why a Simulator

Driving a car in real life is very different to using a simulator. However, there are certain similarities in the content of the images, which allow us to use a simulator as a first approximation to the problem.

The simulator also affords safety and ease-of-use. Data collection is much simpler, and a failed model poses no threat to life. A simulator is a great platform in which to explore and hone various model architectures. A successful model might afterwards be implemented in a real car with real cameras.

The simulator contains
* a **training mode** to record images and steering angles while driving the car around two different tracks
* and an **autonomous mode** that accepts steering angles from a trained model and drives the car based on the prediction.

### The goals/steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### General considerations
The car is equipped with three cameras, one to the left, one in the center and one to the right of the vehicle that provide images from these different view points. It is thus crucial that the CNN does not merely memorize the first track, but generalizes to unseen data in order to perform well on the test track.

The main problem lies in the skew and bias of the data set. The left-right skew is less problematic and can be eliminated by flipping images and steering angles simultaneously. To reduce bias, we have selected the images in the generator based on the probabilities calculated using the function "scipy.stats.norm". This provides more homogeneous training data.

## Rubric Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project contains the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```
#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The architecture used is [Nvidia's CNN architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) that consists of 9 layers, including one normalization layer, 5 convolutional layers and 3 fully connected layers.

The input tensor size for CNN is 66x200x3 and the architecture for convolutional (conv) layers is identical to NVIDIA architecture. We use 24,36 and 48 5x5 filters for the first three conv layers with strides of 2x2. The next two conv layers use 64 filters of size 3x3 with single strides. This is our base model upon which we build our final architecture.

The model includes ELU layers to introduce non-linearity

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               115300    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0
_________________________________________________________________
```

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

I tried to drive most of the time in the center lane. It was easy on the first track, but on the second track it was a bit difficult.

#### 5. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the CNN used in the Traffic Sign Recognition project. I thought this model might be appropriate as an first try because it was something easy to develop.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that dataset could be more homogeneous. Also I applied somo random distort to the images to have more data.

Then, I decided to implement the Nvidia's CNN architecture because I realized that it was impossible to have good results with my previous model

The final step was to run the simulator to see how well the car was driving around track one. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
