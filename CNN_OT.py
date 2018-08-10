#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@autor: Oleksandr Trunov
"""
# Sequential is init of ANN (we can init as graph or sequence of layers)
from keras.models import Sequential
# 2D for pictures and 3D for videos
from keras.layers import Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import regularizers
import numpy as np
import os

class SimpleCNN ():

    def __init__ (self):
        self.classifier = None
        self.classifier_summary = None
        self.callbacks_list = None
        self.colored = None
        self.input_size = None
        self.train_set = None
        self.test_set = None
        self.nb_test_set = None
        self.nb_train_set = None
        self.batch_size = None
    
    def preparing (self, save_filepath_model, process_visualization, min_diff,
                   epochs_check):
        # fix random seed for reproducibility
        np.random.seed(1)
        self.callbacks(save_filepath_model, process_visualization, min_diff, epochs_check)

    def callbacks (self, save_filepath_model, process_visualization, min_diff, epochs_check):
        checkpoint = ModelCheckpoint(save_filepath_model, 
                                     monitor = 'val_acc', 
                                     verbose = process_visualization,
                                     save_best_only = True, 
                                     mode = 'max')
        
        early_stopping = EarlyStopping(monitor = 'val_loss', 
                                       min_delta = min_diff,
                                       patience = epochs_check, 
                                       verbose = process_visualization, 
                                       mode = 'min')
        # For visualisation enter in terminal: tensorboard --logdir path_to_current_dir/Graph
        # Check in browswer: http://localhost:6006
        tensor_board = TensorBoard(log_dir = './Graph',
                                   histogram_freq=0,  
                                   write_graph=True,
                                   write_images=True)
        
        self.callbacks_list = [checkpoint, early_stopping, tensor_board]

    # Target size is the same how in Convolution layer as input shape
    def generate_data_sets (self, path_train, path_test, target_size, batch_size, 
                            class_mode, shuffle, rotation_grad, width_range, 
                            height_range, shear_range, zoom, horizontal_flip, 
                            vertical_flip, whitening):
        
        self.batch_size = batch_size
        # Fitting the CNN to the images
        # Image Data Generator from Keras (Batch generator) - reduce overfitting
        train_datagen = ImageDataGenerator(rotation_range = 40, 
                                           width_shift_range = 0.2,
                                           height_shift_range = height_range,
                                           rescale = 1./255,
                                           shear_range = shear_range,
                                           zoom_range = zoom, 
                                           horizontal_flip = horizontal_flip, 
                                           vertical_flip = vertical_flip,
                                           fill_mode = 'nearest')
        self.input_size = target_size
        self.train_set = train_datagen.flow_from_directory(path_train,
                                                      target_size = target_size, 
                                                      batch_size = batch_size, 
                                                      class_mode = class_mode, 
                                                      shuffle = shuffle)
        
        self.nb_train_set = self.check_number_of_files(path_train)
        
        test_datagen = ImageDataGenerator(rescale = 1./255)
        self.test_set = test_datagen.flow_from_directory(path_test,
                                                      target_size = target_size, 
                                                      batch_size = batch_size, 
                                                      class_mode = class_mode, 
                                                      shuffle = shuffle)
        
        self.nb_test_set = self.check_number_of_files(path_test)
    
    def check_number_of_files (self, path):
        onlyfiles = next(os.walk(path))[2]
        return len(onlyfiles)

    def convert_input_shape (self):
        if self.colored == True:
            x = list(self.input_size)
            x.append(3)
            self.input_size = tuple(x)
        else:
            x = list(self.input_size)
            x.append(1)
            self.input_size = tuple(x)
    

    def add_convolution_and_pooling_layers (self, nb_output_filter, output_size,
                                            pooling_size):
        self.convert_input_shape()
        # Convolution layer (Image x Feature Detector = Feature Map), relu because it is not a linear problem
        self.classifier.add(Conv2D(nb_output_filter,
                              output_size,
                              padding = "same", 
                              data_format = "channels_last", 
                              input_shape = self.input_size,
                              activation = "relu",
                              kernel_regularizer=regularizers.l2(0.01),
                              activity_regularizer=regularizers.l1(0.01)))
        # Max Pooling (take max in sq, you can use average or min too)
        self.classifier.add(MaxPooling2D(pool_size = pooling_size))
        self.classifier.add(BatchNornmalization())
    
    def add_full_connection_layer (self, dropout, nb_input):
        self.classifier.add(Dense(units = nb_input, activation = "relu",
                                  kernel_initializer="uniform",
                                  kernel_regularizer = regularizers.l2(0.01),
                                  activity_regularizer = regularizers.l1(0.01)))
        self.classifier.add(Dropout(dropout))

    def model_building (self, nb_output_filter, output_size, pooling_size, 
                        nb_convolution_pooling_layers, coeff_filter, 
                        nb_hidden_layer, dropout, dropout_coeff, 
                        nb_input_nodes, nb_input_nodes_coeff,
                        model_optimizer = "adam",
                        cost_function = "binary_crossentropy"):
        
        self.classifier = Sequential()
        self.add_convolution_and_pooling_layers(nb_output_filter,
                                                output_size,
                                                pooling_size)
        
        for x in range(1, nb_convolution_pooling_layers):
            self.add_convolution_and_pooling_layers(nb_output_filter*coeff_filter, 
                                                    output_size,
                                                    pooling_size)
        # Flattening (feature maps in vector)
        self.classifier.add(Flatten())
        # Input layer
        self.add_full_connection_layer(dropout, nb_input_nodes)
        # Hidden layer
        for x in range (0, nb_hidden_layer):
            self.add_full_connection_layer(dropout/dropout_coeff,
                                      nb_input_nodes*nb_input_nodes_coeff)
        # Output layer (activation will be softmax by more then 2 output-categories)
        self.classifier.add(Dense(units = 1, activation = "sigmoid"))
        
        # Compiling (if more than 2 classes -> classifier crossentropy)
        self.classifier.compile(optimizer = model_optimizer,
                                loss = cost_function,
                                metrics = ["accuracy"])
        
        self.classifier_summary = self.classifier.summary()

    def model_training (self, nb_epochs):
        self.classifier.fit_generator(self.train_set, 
                                      steps_per_epoch = self.nb_train_set/self.batch_size,
                                      epochs = nb_epochs,
                                      validation_data = self.test_set,
                                      validation_steps = self.nb_test_set/self.batch_size,
                                      callbacks=self.callbacks_list)
    def get_metrics (self):
        scores = self.classifier.evaluate(X, Y, verbose=0)
        print("Model metric :: " + "%s: %.2f%%" % (self.classifier.metrics_names[1], scores[1]*100))

    """
    @param path_train:              path to training files
    @param path_test:               path to testing files
    @param save_filepath_model:     path to save of model
    @param colored:                 if pictures are colored - True (default: "True")
    @param process_visualization:   if the process has to be visualizated - enter 1 as integer (default: 1)
    @param min_diff:                minimum change in the monitored quantity to qualify as an improvement (default: 0.01)
    @param epochs_check:            number of epochs with no improvement after which training will be stopped (default: 2)
    @param target_size:             dimensions to which all images found will be resized (height, width), 
                                    it will be used in convolution layer too (default: (64, 64))
    @param batch_size:              number of batches (default: 32)
    @param class_mode:              set “binary” if you have only two classes to predict, if not set to “categorical” (default: "input")
    @param shuffle:                 whether to shuffle the data (default: True)
    @param rotation_grad:           degree range for random rotations ()
    @param width_range:             ranges (as a fraction of total width) within which to randomly translate 
                                    pictures vertically or horizontally (default: 0.2)
    @param height_range:            ranges (as a fraction of total height) within which to randomly translate 
                                    pictures vertically or horizontally (default: 0.2)
    @param shear_range:             shear angle in counter-clockwise direction in degrees (default: 0.2)
    @param zoom:                    zooming factor of picture (default: 0.2)
    @param horizontal_flip:         randomly flipping half of the images horizontally - True or False (default: True)
    @param vertical_flip:           randomly flipping half of the images vertically - True or False (default: False)
    @param whitening:               whitening of images (default: True)
    @param nb_output_filter:        number of output filter (feature detectors) for convolution (default: 32)
    @param output_size:             size of the output feature map (default: (3, 3))
    @param pooling_size:            size of pooling matrix (default: (2, 2))
    @param nb_convolution_pooling_layers: number of layers -> 
                                    1 layer is 1 convolution and 1 pooling layer (default: 1 (1 convolution and 1 pooling))
    @param coeff_filter:            changing coefficient of nb_output_filter for each layer (default: 1)
    @param nb_hidden_layer:         number of layers of ANN (default: 1)
    @param dropout:                 how much neurons will be droped in training (default: 0.2)
    @param dropout_coeff:           changing coefficient of dropout for each layer (default: 1)
    @param nb_input_nodes:          number of input nodes in ANN (default: nb_output_filter*2)
    @param nb_input_nodes_coeff:    hanging coefficient of nodes in ANN (default: 1)
    @param model_optimizer:         model optimizer, recomend to use adam or RMSprop (default: "adam")
    @param cost_function:           loss or cost function for backpropagation, recomend to use categorical_crossentropy
                                    if you have more then 2 classes and binary_crossentropy if you have only 2 classes (default: "binary_crossentropy")
    @param nb_epochs:               number of epochs (default: 25)
    """
    def creating_new_model(self, path_train,
                                 path_test,
                                 save_filepath_model,
                                 colored = True,
                                 process_visualization = 1,
                                 min_diff = 0.01,
                                 epochs_check = 2,
                                 target_size = (64, 64),
                                 batch_size = 32, 
                                 class_mode = "input",
                                 shuffle = True,
                                 rotation_grad = 90, 
                                 width_range = 0.2, 
                                 height_range = 0.2, 
                                 shear_range = 0.2, 
                                 zoom = 0.2, 
                                 horizontal_flip = True, 
                                 vertical_flip = False, 
                                 whitening = True,
                                 nb_output_filter = 32, 
                                 output_size = (3, 3), 
                                 pooling_size = (2, 2), 
                                 nb_convolution_pooling_layers = 1, 
                                 coeff_filter = 1, 
                                 nb_hidden_layer = 1, 
                                 dropout = 0.2, 
                                 dropout_coeff = 1, 
                                 nb_input_nodes = 0,
                                 nb_input_nodes_coeff = 1,
                                 model_optimizer = "adam",
                                 cost_function = "binary_crossentropy",
                                 nb_epochs = 25):
        
        self.colored = colored
        self.preparing(save_filepath_model, process_visualization, min_diff,
                       epochs_check)
        self.generate_data_sets(path_train,
                            path_test,
                            target_size,
                            batch_size, 
                            class_mode,
                            shuffle,
                            rotation_grad, 
                            width_range, 
                            height_range, 
                            shear_range, 
                            zoom, 
                            horizontal_flip, 
                            vertical_flip, 
                            whitening)
        
        if nb_input_nodes == 0:
            nb_input_nodes = nb_output_filter*2
            
        self.model_building(nb_output_filter = nb_output_filter, 
                            output_size = output_size, 
                            pooling_size = pooling_size, 
                            nb_convolution_pooling_layers = nb_convolution_pooling_layers, 
                            coeff_filter = coeff_filter, 
                            nb_hidden_layer = nb_hidden_layer, 
                            dropout = dropout, 
                            dropout_coeff = dropout_coeff, 
                            nb_input_nodes = nb_input_nodes,
                            nb_input_nodes_coeff = nb_input_nodes_coeff,
                            model_optimizer = model_optimizer,
                            cost_function = cost_function)
        self.model_training(nb_epochs = nb_epochs)
        self.classifier.save("my_model.h5")

def main():
    cnn = SimpleCNN()
    cnn.creating_new_model(colored = True,
                                 save_filepath_model = "./my_model_.h5",
                                 process_visualization = 0,
                                 min_diff = 0.01,
                                 epochs_check = 2,
                                 path_train = 'dataset/training_set',
                                 path_test = 'dataset/test_set',
                                 target_size = (64, 64),
                                 batch_size = 32, 
                                 class_mode = "binary",
                                 shuffle = True,
                                 rotation_grad = 90, 
                                 width_range = 0.2, 
                                 height_range = 0.2, 
                                 shear_range = 0.2, 
                                 zoom = 0.2, 
                                 horizontal_flip = True, 
                                 vertical_flip = False, 
                                 whitening = True,
                                 nb_output_filter = 32, 
                                 output_size = (3, 3), 
                                 pooling_size = (2, 2), 
                                 nb_convolution_pooling_layers = 1, 
                                 coeff_filter = 1, 
                                 nb_hidden_layer = 1, 
                                 dropout = 0.2, 
                                 dropout_coeff = 2, 
                                 nb_input_nodes = 0,
                                 nb_input_nodes_coeff = 2,
                                 model_optimizer = "adam",
                                 cost_function = "binary_crossentropy",
                                 nb_epochs = 50)

if __name__ == "__main__":
    main()