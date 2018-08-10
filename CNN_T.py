#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Oleksandr Trunov
"""
# Sequential is init of ANN (we can init as graph or sequence of layers)
from keras.models import Sequential
# 2D for pictures and 3D for videos
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import regularizers
import numpy as np
import os

class CNN ():
    
    def __init__ (self):
        self.nb_train_set = 0
        self.nb_test_set = 0
        self.classifier = None
        self.colored = True
        self.target_size = None
        self.classifier_summary = None
        self.batch_size = 0
        # fix random seed for reproducibility
        np.random.seed(1)

    # Private Function
    # Callback if accurancy or loss will be sunken or increased
    # Visualisation with TensorBoard (see comments below)
    def callbacks (self):
        checkpoint = ModelCheckpoint("./CNN_model_checkpoint.h5", 
                                     monitor = 'val_acc', 
                                     verbose = 0,
                                     save_best_only = True, 
                                     mode = 'max')
            
        early_stopping = EarlyStopping(monitor = 'val_loss', 
                                       min_delta = 0.01,
                                       patience = 5, 
                                       verbose = 0, 
                                       mode = 'min')
        # For visualisation enter in terminal: tensorboard --logdir path_to_current_dir/Graph
        # Check in browswer: http://localhost:6006
        tensor_board = TensorBoard(log_dir = './Graph',
                                   histogram_freq=0,  
                                   write_graph=True,
                                   write_images=True)
            
        callbacks_list = [checkpoint, early_stopping, tensor_board]
        return callbacks_list
    
    def generate_train_set (self, path_train, target_size, batch_size, class_mode):
        train_datagen = ImageDataGenerator(rotation_range = 45, 
                                           width_shift_range = 0.2,
                                           height_shift_range = 0.2,
                                           rescale = 1./255,
                                           shear_range = 0.2,
                                           zoom_range = 0.2, 
                                           horizontal_flip = True, 
                                           vertical_flip = True,
                                           fill_mode = 'nearest')
        self.batch_size = batch_size
        train_set = train_datagen.flow_from_directory(path_train,
                                                      target_size = target_size, 
                                                      batch_size = batch_size, 
                                                      class_mode = class_mode,
                                                      shuffle = True)
            
        self.nb_train_set = self.check_number_of_files(path_train)
        return train_set
            
    def generate_test_set (self, path_test, target_size, batch_size, class_mode):
        test_datagen = ImageDataGenerator(rescale = 1./255)
        test_set = test_datagen.flow_from_directory(path_test,
                                                    target_size = target_size, 
                                                    batch_size = batch_size, 
                                                    class_mode = class_mode,
                                                    shuffle = True)
            
        self.nb_test_set = self.check_number_of_files(path_test)
        return test_set
    
    # Private function
    # Check number of files for training and test sets
    def check_number_of_files (self, path):
        onlyfiles = next(os.walk(path))[2]
        return len(onlyfiles)
    # Private function
    # Converting of shapes, depend on the colored (RGB) or b/w pictures
    def convert_input_shape (self):
        if self.colored == True:
            x = list(self.target_size)
            x.append(3)
            self.target_size = tuple(x)
        else:
            x = list(self.target_size)
            x.append(1)
            self.target_size = tuple(x)
    
    def add_convolution_and_pooling_layers (self, nb_convolution_filter, output_size,
                                            pooling_size):
        self.convert_input_shape()
        # Convolution layer (Image x Feature Detector = Feature Map), relu because it is not a linear problem
        self.classifier.add(Conv2D(nb_convolution_filter,
                                   output_size,
                                   padding = "same", 
                                   data_format = "channels_last", 
                                   input_shape = self.target_size,
                                   activation = "relu",
                                   kernel_regularizer=regularizers.l2(0.01),
                                   activity_regularizer=regularizers.l1(0.01)))
        # Max Pooling (take max in sq, you can use average or min too)
        self.classifier.add(MaxPooling2D(pool_size = pooling_size))
        self.classifier.add(BatchNormalization())
     
    def add_full_connection_layer (self, dropout, nb_input_nodes):
        self.classifier.add(Dense(units = nb_input_nodes,
                             activation = "relu",
                             kernel_initializer="uniform",
                             kernel_regularizer = regularizers.l2(0.01),
                             activity_regularizer = regularizers.l1(0.01)))
        self.classifier.add(Dropout(dropout))
    
    def model_building (self, nb_convolution_filter, kernal_matrix_size, pooling_size, 
                        nb_convolution_pooling_layers, coeff_filter, 
                        nb_hidden_layer, dropout, dropout_coeff, 
                        nb_input_nodes, nb_input_nodes_coeff,
                        nb_classes, nb_classes_function, cost_function):
            
        self.classifier = Sequential()
        self.add_convolution_and_pooling_layers(nb_convolution_filter, kernal_matrix_size, pooling_size)
        for x in range(1, nb_convolution_pooling_layers):
            self.add_convolution_and_pooling_layers(nb_convolution_filter*coeff_filter*x, kernal_matrix_size, pooling_size)
        # Flattening (feature maps in vector)
        self.classifier.add(Flatten())
        # Input layer
        self.add_full_connection_layer(dropout, nb_input_nodes)
        # Hidden layer
        for x in range (0, nb_hidden_layer):
            self.add_full_connection_layer(dropout/dropout_coeff,
                                          nb_input_nodes*nb_input_nodes_coeff*x)
        # Output layer (activation will be softmax by more then 2 output-categories)
        self.classifier.add(Dense(units = nb_classes, activation = nb_classes_function))
            
        # Compiling (if more than 2 classes -> classifier crossentropy)
        self.classifier.compile(optimizer = "adam", loss = cost_function, metrics = ["accuracy"])
        self.classifier_summary = self.classifier.summary()
    
    def model_training (self, nb_epochs, path_train, path_test, target_size, batch_size, class_mode):
        self.classifier.fit_generator(self.generate_train_set(path_train, target_size, batch_size, class_mode), 
                                      steps_per_epoch = self.nb_train_set/self.batch_size,
                                      epochs = nb_epochs,
                                      validation_data = self.generate_test_set(path_test, target_size, batch_size, class_mode),
                                      validation_steps = self.nb_test_set/self.batch_size,
                                      callbacks=self.callbacks())
    
    def creating_new_model(self, 
                           path_train,
                           path_test,
                           colored,
                           nb_convolution_filter,
                           kernal_matrix_size,
                           pooling_size,
                           nb_convolution_pooling_layers,
                           coeff_filter,
                           nb_hidden_layer,
                           dropout,
                           dropout_coeff,
                           nb_input_nodes,
                           nb_input_nodes_coeff,
                           nb_epochs,
                           target_size,
                           batch_size,
                           class_mode,
                           nb_classes,
                           nb_classes_function = "sigmoid",
                           cost_function = "binary_crossentropy"):
        self.target_size = target_size
        self.colored = colored
        self.model_building (nb_convolution_filter, kernal_matrix_size, pooling_size, 
                             nb_convolution_pooling_layers, coeff_filter, 
                             nb_hidden_layer, dropout, dropout_coeff, 
                             nb_input_nodes, nb_input_nodes_coeff,
                             nb_classes, nb_classes_function,
                             cost_function)
        self.model_training (nb_epochs, path_train, path_test, target_size, batch_size, class_mode)
        self.classifier.save("CNN_model_last.h5")

def main():
    cnn = CNN()
    cnn.creating_new_model(path_train = 'dataset/training_set',
                           path_test = 'dataset/test_set',
                           colored = True,
                           nb_convolution_filter = 32,
                           kernal_matrix_size = (3, 3),
                           pooling_size = (2, 2),
                           nb_convolution_pooling_layers = 1,
                           coeff_filter = 2,
                           nb_hidden_layer = 2,
                           dropout = 0.2,
                           dropout_coeff = 2,
                           nb_input_nodes = 64,
                           nb_input_nodes_coeff = 2,
                           nb_classes = 1,
                           nb_classes_function = "sigmoid",
                           cost_function = "binary_crossentropy",
                           nb_epochs = 25,
                           target_size = (64, 64),
                           batch_size = 32,
                           class_mode = "binary")

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    # Make predictions
#    import numpy as np
#    from keras.preprocessing import image
#    
#    test = image.load_img("prediction_img.jpg", target_size = (64, 64))
#    # Make 3D array (RGB) - because we have "input_shape = (64, 64, 3)"
#    test = image.img_to_array(test)
#    # Error when checking input: expected conv2d_3_input to have 4 dimensions, but got array with shape (64, 64, 3)
#    # because:
#    test = np.expand_dims(test, axis = 0)
#    result = classifier.predict(test)
#    # Mapping classes and result
#    index = train_set.class_indices
#    if result[0][0] == 1:
#        prediction = "This is class1"
#    else:
#        prediction = "This is class2"