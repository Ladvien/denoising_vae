#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:27:59 2020

@author: ladvien
"""

import cv2

import sys
import os

import matplotlib.pyplot as plt

image_tools_path = "/home/ladvien/deep_arcane/"
sys.path.append(image_tools_path)

from image_utils import ImageUtils
iu = ImageUtils()



#############
# Parameters
#############

input_path = "/home/ladvien/denoising_vae/data/extracted/"
output_path = "/home/ladvien/deep_arcane/6_vae_denoiser/data/"

threshold = 240

#############
# Extract
#############

clear_img_path = f"{output_path}clear/"
noise_img_path = f"{output_path}nose/"


if not os.path.exists(clear_img_path):
    os.makedirs(clear_img_path)

if not os.path.exists(noise_img_path):
    os.makedirs(noise_img_path)

file_paths = iu.get_image_files_recursively(input_path)

counter = 0
for file_path in file_paths:
    file_name = file_path.split("/")[-1]
    outpout_file_path = output_path + file_name
    
    image = cv2.imread(file_path)
    
    
    # ADD NOISE HERE
    
        
    try:
        print(f"Writing file {outpout_file_path}")
        cv2.imwrite(outpout_file_path, image)
    except:
        print(f"Removed: {file_path}")
