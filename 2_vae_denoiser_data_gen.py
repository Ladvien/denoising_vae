#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:27:59 2020

@author: ladvien
"""


import sys
import os

import cv2
import numpy as np
from random import randint

import matplotlib.pyplot as plt

image_tools_path = "/home/ladvien/deep_arcane/"
sys.path.append(image_tools_path)

from image_utils import ImageUtils
iu = ImageUtils()



def noisy(noise_typ, image):
    if noise_typ == "gauss":
       row,col,ch= image.shape
       mean = 0
       var = 0.1
       sigma = var**0.5
       gauss = np.random.normal(mean,sigma,(row,col,ch))
       gauss = gauss.reshape(row,col,ch)
       noisy = image + gauss
       return noisy
    elif noise_typ == "s&p":
       row,col,ch = image.shape
       s_vs_p = 0.5
       amount = 0.004
       out = np.copy(image)
       # Salt mode
       num_salt = np.ceil(amount * image.size * s_vs_p)
       coords = [np.random.randint(0, i - 1, int(num_salt))
               for i in image.shape]
       out[coords] = 1
    
       # Pepper mode
       num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
       coords = [np.random.randint(0, i - 1, int(num_pepper))
               for i in image.shape]
       out[coords] = 0
       return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

#############
# Parameters
#############

input_path = "/home/ladvien/denoising_vae/data/extracted/"
output_path = "/home/ladvien/denoising_vae/data/training/"

threshold               = 240
color_range             = 30
shape_range             = 15
size_range              = 1
num_pepper              = 20
specks_per_pepper       = 60
group_range             = 12

image_shape             = (128, 128)

show                    = False

#############
# Extract
#############

clear_img_path = f"{output_path}clear/"
noise_img_path = f"{output_path}noise/"


if not os.path.exists(clear_img_path):
    os.makedirs(clear_img_path)

if not os.path.exists(noise_img_path):
    os.makedirs(noise_img_path)

file_paths = iu.get_image_files_recursively(input_path)

counter = 0
for file_path in file_paths:
    file_name = file_path.split("/")[-1]
    outpout_file_path = output_path + file_name
    
    clear_image = cv2.imread(file_path)
    noise_img = clear_image.copy()
     
    for i in range(0, num_pepper):
        # Radius of circle
        radius = randint(0, shape_range)
          
        b = randint(0, color_range)
        g = randint(0, color_range)
        r = randint(0, color_range)
        
        # BGR
        color = (b, g, r)
    
        # Center coordinates
        y = randint(0, image_shape[1])    
        x = randint(0, image_shape[1])
        
        for j in range(0, specks_per_pepper):        
        
            group_x_offset = randint(group_range*-1, group_range)
            group_y_offset = randint(group_range*-1, group_range)
            
            # Size
            radius = randint(0, size_range)
            noise_img = cv2.circle(noise_img, (x + group_x_offset, y + group_y_offset), radius, color, -1)
     
    if show:
        plt.imshow(noise_img, cmap="gray")
        plt.show()
        
    try:
        file_name = f"{counter}.png"
        print(f"Writing file {file_name}")
        cv2.imwrite(noise_img_path + file_name, noise_img)        
        cv2.imwrite(clear_img_path + file_name, clear_image)
    except:
        print(f"Removed: {file_path}")

    counter+=1
