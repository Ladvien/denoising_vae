#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:25:42 2020

@author: ladvien
"""


# Import Tensorflow
import tensorflow as tf

# Import needed tools.
import os
import sys
import matplotlib.pyplot as plt
import json
import numpy as np
from scipy import stats

# Viewing
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Import Keras
import tensorflow.keras
from tensorflow.keras.layers import Dense,Flatten, Dropout, Lambda
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, MaxPooling2D, Conv2D, Activation
from tensorflow.compat.v1.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image

# import cv2


image_tools_path = "/home/ladvien/deep_arcane/"
sys.path.append(image_tools_path)

from image_utils import ImageUtils
iu = ImageUtils()


#################################
# TODO: Make experiment folder
#################################
# 1. Generate a unique id
# 2. Save weighs to folder
# 3. Save tensorboard logs and open tensorboard to this folder

#################################
# Training Parameters
#################################

root_path               = "/home/ladvien/deep_arcane/4_bold_symbol_classifier"

continue_training       = False
initial_epoch           = 0
clear_logs              = True

input_shape             = (128, 128, 1) # This is the shape of the image width, length, colors
image_size              = (input_shape[0], input_shape[1]) # DOH! image_size is (height, width)
train_test_ratio        = 0.2
zoom_range              = 0.0
shear_range             = 0.0

# Hyperparameters
batch_size              = 32
epochs                  = 30
steps_per_epoch         = 300
validation_steps        = 50 
optimizer               = 'adam' 
learning_rate           = 0.001
val_save_step_num       = 1
dropout                 = 0.15

path_to_graphs          = f'{root_path}/data/output/logs/'
model_save_dir          = f'{root_path}/data/output/'
train_dir               = f'{root_path}/train/'
val_dir                 = f'{root_path}/test/'

#################################
# Helper functions
#################################

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    

def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history['loss'], label='Train loss')
    ax[0].plot(history.epoch, history.history['val_loss'], label='Validation loss')
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history['acc'], label='Train acc')
    ax[1].plot(history.epoch, history.history['val_acc'], label='Validation acc')
    ax[0].legend()
    ax[1].legend()
    
#################################
# Create needed dirs
#################################
make_dir(path_to_graphs)
make_dir(model_save_dir)

#################################
# Data generators
#################################

# These Keras generators will pull files from disk
# and prepare them for training and validation.

# Determine color depth.
color_mode = ''
if input_shape[2] == 1:
    color_mode = 'grayscale'
elif input_shape[2] == 3:
    color_mode = 'rgb'
elif input_shape[2] == 4:
    color_mode = 'rgba'

print(f'Color mode: {color_mode}')

augs_gen = ImageDataGenerator (
    shear_range = shear_range,
    brightness_range = [0.9, 1.1],
    rotation_range = 70,
    # width_shift_range = 0.1,
    # height_shift_range = 0.1,
    zoom_range = zoom_range,        
    horizontal_flip = True,
    validation_split = train_test_ratio,
    fill_mode = 'nearest'
)  

train_gen = augs_gen.flow_from_directory (
    train_dir,
    target_size = image_size, # THIS IS HEIGHT, WIDTH
    batch_size = batch_size,
    class_mode = 'binary',
    shuffle = True,
    color_mode = color_mode
)

test_gen = augs_gen.flow_from_directory (
    val_dir,
    target_size = image_size,
    batch_size = batch_size,
    class_mode = 'binary',
    shuffle = False,
    color_mode = color_mode
)

#################################
# Check Generator Output
#################################
x_batch, y_batch = next(test_gen)
for i in range(x_batch.shape[0]):
    plt.imshow(x_batch[i].astype('uint8'))
    plt.show()

#################################
# Save Class IDs
#################################
classes_json = train_gen.class_indices
num_classes = len(train_gen.class_indices)

with open(model_save_dir + 'classes.json', 'w') as fp:
    json.dump(classes_json, fp, indent = 4)

#################################
# Model Building
#################################

def test_model(opt, input_shape, dropout = 0.0):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape = input_shape))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tf.keras.layers.Conv2D(128, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tf.keras.layers.Conv2D(512, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tf.keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Activation('relu'))
    
    model.add(tf.keras.layers.Dropout(dropout))
    
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))
    
    return model

#################################
# Create model
#################################

def get_optimizer(optimizer, learning_rate = 0.001):
    if optimizer == 'adam':
        return tensorflow.keras.optimizers.Adam(lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0., amsgrad = False)
    elif optimizer == 'sgd':
        return tensorflow.keras.optimizers.SGD(lr = learning_rate, momentum = 0.99) 
    elif optimizer == 'adadelta':
        return tensorflow.keras.optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=None, decay=0.0)

selected_optimizer = get_optimizer(optimizer, learning_rate)

model = test_model(selected_optimizer, input_shape)
model.summary()

model.compile(
    loss = 'binary_crossentropy',
    optimizer = selected_optimizer,
    metrics = ['accuracy']
)

#################################
# Keras Callbacks
#################################
best_model_weights = model_save_dir + 'model.h5'

checkpoint = ModelCheckpoint(
    best_model_weights,
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True,
    mode = 'min',
    save_weights_only = False,
    save_freq = val_save_step_num
)

earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=10,
    verbose=1,
    mode='auto'
)

tensorboard = TensorBoard(
    log_dir = model_save_dir + '/logs',
    histogram_freq=0,
    batch_size=16,
    write_graph=True,
    write_grads=True,
    write_images=False,
)

csvlogger = CSVLogger(
    filename = model_save_dir + 'training.csv',
    separator = ',',
    append = False
)

reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=40,
    verbose=1, 
    mode='auto',
    cooldown=1 
)

callbacks = [earlystop]

#################################
# Execute Training
#################################

if continue_training:
    model.load_weights(best_model_weights)
    model_score = model.evaluate_generator(test_gen, steps = validation_steps)

    print('Model Test Loss:', model_score[0])
    print('Model Test Accuracy:', model_score[1])


history = model.fit_generator(
    train_gen, 
    steps_per_epoch  = steps_per_epoch, 
    validation_data  = test_gen,
    validation_steps = validation_steps,
    epochs = epochs, 
    verbose = 1,
    callbacks = callbacks
)

#################################
# Save Model
#################################
model_json = model.to_json()
with open(model_save_dir + 'model.json', 'w') as json_file:
    json_file.write(model_json)
    
model.save(model_save_dir + 'model.h5')
print('Weights Saved')

#################################
# Evaluate Training
#################################
model.load_weights(best_model_weights)
model_score = model.evaluate_generator(test_gen, steps = validation_steps)

print('Model Test Loss:', model_score[0])
print('Model Test Accuracy:', model_score[1])
    


#################################
# Test Image
#################################

# 1. Get each class and label.
# 2. Generate nubmber of n predictions for each class.
# 3. Take the mode of the predictions of each class.
# 4. Compare the prediction mode against actual class.



rev_lookup = dict([[v,k] for k,v in classes_json.items()])

input_path = "/home/ladvien/deep_arcane/images/0_raw/6_resized/"
file_paths = iu.get_image_files_recursively(input_path)

sorted_path = "/home/ladvien/deep_arcane/images/0_raw/sorted"
symbol_path = f"{sorted_path}/bold_symbol/"
non_symbol_path = f"{sorted_path}/non_bold_symbol/"
    
make_dir(symbol_path)
make_dir(non_symbol_path)

for file_path in file_paths:
    
    print(f"Classifying {file_path}")
    filename = file_path.split("/")[-1]
    
    x = image.load_img(file_path, grayscale=True, target_size = input_shape)
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    result = model.predict_classes(x, batch_size=1)[0][0]
    
    
    if result > 0:
        os.system(f"cp {file_path} {symbol_path}{filename}")
    else:
        os.system(f"cp {file_path} {non_symbol_path}{filename}")
