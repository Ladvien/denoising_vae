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
from random import randint
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

from PIL import Image


image_tools_path = "/home/ladvien/deep_arcane/"
sys.path.append(image_tools_path)

from image_utils import ImageUtils
iu = ImageUtils()


#################################
# TODO: Make experiment folder
#################################


#################################
# Training Parameters
#################################

root_path               = "/home/ladvien/denoising_vae"

continue_training       = False
initial_epoch           = 0
clear_logs              = True

input_shape             = (128, 128, 1) # This is the shape of the image width, length, colors
image_size              = (input_shape[0], input_shape[1]) # DOH! image_size is (height, width)
train_test_ratio        = 0.2
zoom_range              = 0.0
shear_range             = 0.0

# Hyperparameters
batch_size              = 64
epochs                  = 300
steps_per_epoch         = 300
validation_steps        = 50 
optimizer               = 'adam' 
learning_rate           = 0.001
val_save_step_num       = 1
dropout                 = 0.0

path_to_graphs          = f'{root_path}/data/output/logs/'
model_save_dir          = f'{root_path}/data/output/'
train_dir               = f'{root_path}/data/train/'
val_dir                 = f'{root_path}/data/test/'

#################################
# Get Train Files
#################################
clear_file_paths = iu.get_image_files_recursively(train_dir + "clear/")


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

#################################
# Model Building
#################################

def test_model(opt, input_shape, dropout = 0.0):
    
    initializer = tf.keras.initializers.GlorotNormal()
    # Encoder
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (3, 3), input_shape = input_shape, padding="same"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Decoder
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.UpSampling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.UpSampling2D((2, 2)))
    
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.UpSampling2D((2, 2)))
    
    model.add(tf.keras.layers.Conv2D(1, (3, 3), padding="same"))
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

best_model_weights = model_save_dir + 'model.h5'


#################################
# Execute Training
#################################

import time
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

train_acc_metric = tf.keras.metrics.BinaryAccuracy()
val_acc_metric = tf.keras.metrics.BinaryAccuracy()


def get_batch(file_paths, batch_size, verbose = 0):
    
    # Batches to return.
    clear_batch = []
    noise_batch = []
    
    # Get random files
    file_nums_to_load = [randint(0, len(file_paths)) - 1 for x in range(0, batch_size)]
    
    # Loop through the batch size, loading files.
    for i in range(0, batch_size):
        
        # Create matching file paths.
        clear_image_file_path = file_paths[file_nums_to_load[i]]
        noise_image_file_path = file_paths[file_nums_to_load[i]].replace("clear", "noise")
        if verbose > 0:
            print(f"Loading clear file: {clear_image_file_path}")
            print(f"Loading noise file: {noise_image_file_path}")
        
        # Load CLEAR image, convert to B&W, and put into training batch.
        clear_image = Image.open(clear_image_file_path).convert("1")
        clear_batch.append(np.array(clear_image, dtype=int))
        
        # Load NOISE image, convert to B&W, and put into training batch.
        noise_image = Image.open(noise_image_file_path).convert("1")
        noise_batch.append(np.array(noise_image, dtype=int))
        
    return (np.array(noise_batch), np.array(clear_batch))

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step in range(0, steps_per_epoch):
        
        # Load training batch.
        x_batch_train, y_batch_train = get_batch(clear_file_paths, batch_size)
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train.reshape(logits.shape), logits)
            
        grads = tape.gradient(loss_value, model.trainable_weights)
        selected_optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
        # Update training metric.
        train_acc_metric.update_state(y_batch_train.reshape(logits.shape), logits)
    
    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print(f"Training loss: {loss_value}")
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # # Run a validation loop at the end of each epoch.
    # for x_batch_val, y_batch_val in val_dataset:
    #     val_logits = model(x_batch_val, training=False)
    #     # Update val metrics
    #     val_acc_metric.update_state(y_batch_val, val_logits)
        
    # val_acc = val_acc_metric.result()
    # val_acc_metric.reset_states()
    # print("Validation acc: %.4f" % (float(val_acc),))
    # print("Time taken: %.2fs" % (time.time() - start_time))


#################################
# Save Model
#################################
model_json = model.to_json()
with open(model_save_dir + 'model.json', 'w') as json_file:
    json_file.write(model_json)
    
model.save(model_save_dir + 'model.h5')
print('Weights Saved')


#################################
# Test Image
#################################

# 1. Get each class and label.
# 2. Generate nubmber of n predictions for each class.
# 3. Take the mode of the predictions of each class.
# 4. Compare the prediction mode against actual class.


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
