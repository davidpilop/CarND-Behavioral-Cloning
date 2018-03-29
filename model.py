"""First we load all training data. We make use of data given by Udacity (lines 7-17) and data collected by myself (lines 19-23)"""

from glob import glob
import pandas
import os
#Data provided by Udacity
dataDIR = '.\data'
colnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data = pandas.read_csv('data/driving_log.csv', skiprows=[0], names=colnames)
center = data.center.tolist()
left = data.left.tolist()
right = data.right.tolist()
steering = data.steering.tolist()

left = glob(os.path.join(dataDIR,'IMG/left*.jpg'))
center = glob(os.path.join(dataDIR,'IMG/center*.jpg'))
right = glob(os.path.join(dataDIR,'IMG/right*.jpg'))
#Data recorded by myself
data = pandas.read_csv('myData/driving_log.csv', skiprows=[0], names=colnames)
center = center + data.center.tolist()
left = left + data.left.tolist()
right = right + data.right.tolist()
steering = steering + data.steering.tolist()
#number of images per camera
nb_img = len(steering)

"""Then, we create a function that returns the image and the steering angle, given the index of the steering angle and the camera used"""

import random
import cv2

cam_positions = ['left','center','right']

#load image according to camera
def generate_image(img_index, cam_pos='center'):
    if cam_pos == 'random':
        cam_pos = np.random.choice(cam_positions)
    if cam_pos == 'center':
        img = cv2.imread(center[img_index])
        steer = steering[img_index]
    elif cam_pos == 'left':
        img = cv2.imread(left[img_index])
        steer = steering[img_index] + 0.2
    elif cam_pos == 'right':
        img = cv2.imread(right[img_index])
        steer = steering[img_index] - 0.2
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return image, steer

"""Here is one of the most important parts of the code. To reduce bias, we have selected the images in the generator based on the probabilities calculated using the function "scipy.stats.norm". This provides more homogeneous training data."""

import numpy as np
from scipy.stats import norm

nb_steer = 25
steer_bins = np.linspace(-1., 1.001, nb_steer+1)
steer_bin_index = list(range(1,nb_steer+1))
data_bin_index = np.digitize(steering, steer_bins)

# selection probability for each bin
steer_bin_prob = np.array([norm.pdf(x, scale=.8) for x in steer_bins])
steer_bin_prob = {i:steer_bin_prob[i-1]/np.sum(steer_bin_prob) for i in steer_bin_index}

# selection probability for each image
img_prob = np.zeros(nb_img)

data_steer_bins = {steer_bin : [] for steer_bin in steer_bin_index}
for i in range(nb_img) :
    key = data_bin_index[i]
    data_steer_bins[key].append(i)

for index in steer_bin_index :
    bin_size = len(data_steer_bins[index])
    p_bin = steer_bin_prob[index]
    for i in data_steer_bins[index] :
        img_prob[i] = p_bin/bin_size

input_shape = (66,200,3)

def crop_img(img, top=30, bottom=24, left=0, right=0) :
    h,w,_ = img.shape
    return img[top:h-bottom,left:w-right]

def AbsNorm(image, a=-.5, b=0.5, col_min=0, col_max=255) :
    return (image-col_min)*(b-a)/(col_max-col_min)

def contrast_norm(image) :
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    new_image[:,:,0] = cv2.equalizeHist(new_image[:,:,0])
    new_image = cv2.cvtColor(new_image, cv2.COLOR_YUV2BGR)
    return AbsNorm(new_image)

def preprocess(img) :
    new_img = crop_img(img)
    new_img = contrast_norm(new_img)
    new_img = cv2.resize(new_img, (200,66), interpolation = cv2.INTER_AREA)
    return new_img

delta_steer = {'left':0.2, 'center':0., 'right':-0.2}

def random_distort(image, steer, shift=0.4, rotation=0.4):
    delta1 = 70*shift/delta_steer['left'] # calibration
    delta2 = 70*rotation/delta_steer['left'] # calibration
    h,w,_ = image.shape
    pts1 = np.float32([[w/2-30,h/3],[w/2+30,h/3],[w/2-80,h],[w/2+80,h]])
    pts2 = np.float32([[w/2-30+delta2,h/3],[w/2+30+delta2,h/3],[w/2-80-delta1,h],[w/2+80-delta1,h]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    image = cv2.warpPerspective(image,M,(w,h))
    steer += -shift + rotation
    return image, steer

def generator(data_indices, batch_size=64, train=False) :
    start_index = None if train else 0

    X_batch = np.zeros((batch_size, *input_shape))
    y_batch = np.zeros(batch_size)

    # infinite loop for generator
    while True :
        batch_indices = None
        if train :
            prob =  img_prob[data_indices]/np.sum(img_prob[data_indices])
            batch_indices = np.random.choice(data_indices, batch_size, p=prob)
        else :
            batch_indices = np.random.choice(data_indices, batch_size)

        images = []
        angles = []
        # generate images
        for i,img_index in enumerate(batch_indices):
            img = None
            if train :
                img, steer = generate_image(img_index, cam_pos='random')
                #Flip image
                img_flip = cv2.flip(img,1)
                images.append(preprocess(img_flip))
                angles.append(-steer)
                #distort image
                img_distort, steer_distort = random_distort(img, steer,
                                            shift=random.uniform(delta_steer['right']/2,delta_steer['left']/2),
                                            rotation=random.uniform(delta_steer['right']/2,delta_steer['left']/2))
                images.append(preprocess(img_distort))
                angles.append(steer_distort)
                img_distort_flip = cv2.flip(img_distort,1)
                images.append(preprocess(img_distort_flip))
                angles.append(-steer_distort)
            else :
                img, steer = generate_image(img_index)
            images.append(preprocess(img))
            angles.append(steer)
        inputs = np.array(images)
        outputs = np.array(angles)
        yield sklearn.utils.shuffle(inputs, outputs)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Conv2D
input_shape = (66,200,3)

def nVidiaModel():
    model = Sequential()
    model.add(Conv2D(24, (5, 5), activation="elu", strides=(2, 2), input_shape=input_shape))
    model.add(Conv2D(36, (5, 5), activation="elu", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation="elu", strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

model = nVidiaModel()
model.summary()

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn

# Compiling and training the model
EPOCHS = 10
batch_size = 64

# initialize generators
train_indices, valid_indices = train_test_split(list(range(nb_img)), test_size = 0.2, random_state=0)
train_generator = generator(train_indices, batch_size, train=True)
validation_generator = generator(valid_indices, batch_size)

# Model creation
model = nVidiaModel()

## train model
print('Training ...')
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, epochs = EPOCHS, verbose = 1, validation_data = validation_generator,
                    validation_steps = len(valid_indices)/EPOCHS, steps_per_epoch = len(train_indices)/EPOCHS)

# Save model data
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save("model.h5")
print("Saved model to disk")
