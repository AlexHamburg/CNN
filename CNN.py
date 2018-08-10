#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: CNN with Keras
@author: Oleksandr Trunov
"""
# Building CNN
# Sequential is init of ANN (we can init as graph or sequence of layers)
from keras.models import Sequential
# 2D for pictures and 3D for videos
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
import numpy as np

# fix random seed for reproducibility
np.random.seed(1)


classifier = Sequential()

# Convolution layer (Image x Feature Detector = Feature Map), relu because it is not a linear problem
classifier.add(Conv2D(32, (3, 3), padding = "same", data_format = "channels_last", 
                      input_shape = (64, 64, 3), activation = "relu"))

# Max Pooling (take max in sq, you can use average or min too)
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# For deeper learning can be added additional convolution or full connection layers
# Adding second convolution layer
classifier.add(Conv2D(32, (3, 3), data_format = "channels_last", activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening (feature maps in vector)
classifier.add(Flatten())

# Full Connection
# Hidden layer
classifier.add(Dense(output_dim = 128, activation = "relu"))
classifier.add(Dropout(0.2))

# Output layer (activation will be softmax by more then 2 output-categories)
classifier.add(Dense(output_dim = 1, activation = "sigmoid"))

# Compiling (if more than 2 classes -> classifier crossentropy)
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Fitting the CNN to the images
# Image Data Generator from Keras (Batch generator) - reduce overfitting

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2,
        zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Target size is same how in Convolution layer as input shape
# We have only two classes because class_mode = 'binary'
# For better result we can enter a better target_size, for example (128, 128)
# We can use too: train = pd.read_csv(os.path.join(data_dir, 'Train', 'train.csv'))
# test = pd.read_csv(os.path.join(data_dir, 'Test.csv'))
train_set = train_datagen.flow_from_directory('dataset/training_set',
        target_size = (64, 64), batch_size = 32, class_mode = 'input')

test_set = test_datagen.flow_from_directory('dataset/test_set',
        target_size = (64, 64), batch_size = 32, class_mode = 'binary')

# How much do you have files / pictures in train_set -> steps_per_epoch
# How much do you have files / pictures in test_set -> validation_steps
classifier.fit_generator(train_set, steps_per_epoch = 8000, epochs = 10,
        validation_data = test_set, validation_steps = 2000)

# Save and load of CNN or ANN: https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
classifier.save("my_model.h5")

# Make predictions
import numpy as np
from keras.preprocessing import image

test = image.load_img("prediction_img.jpg", target_size = (64, 64))
# Make 3D array (RGB) - because we have "input_shape = (64, 64, 3)"
test = image.img_to_array(test)
# Error when checking input: expected conv2d_3_input to have 4 dimensions, but got array with shape (64, 64, 3)
# because:
test = np.expand_dims(test, axis = 0)
result = classifier.predict(test)
# Mapping classes and result
index = train_set.class_indices
if result[0][0] == 1:
    prediction = "This is class1"
else:
    prediction = "This is class2"
