#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 19:00:54 2020

@author: ladvien
"""

import cv2

import sys
import os


image_tools_path = "/home/ladvien/deep_arcane/"
sys.path.append(image_tools_path)

from image_utils import ImageUtils
iu = ImageUtils()



#############
# Parameters
#############

input_path = "/home/ladvien/denoising_vae/data/raw/"
output_path = "/home/ladvien/denoising_vae/data/extracted/"

threshold = 240
minimum_size = 30
target_size = (128, 128)

dry_run = False

#############
# Extract
#############

if not os.path.exists(output_path):
    os.makedirs(output_path)

file_paths = iu.get_image_files_recursively(input_path)

index = 0
for file_path in file_paths:
    
    filename = file_path.split("/")[-1].split(".")[0]
    
    iu.save_subimages(filename, file_path, output_path, minimum_size = minimum_size)
    index += 1
    if index > 10 and dry_run:
        break
    

#############
# Resize
#############

file_paths = iu.get_image_files_recursively(output_path)

index = 0
for file_path in file_paths:
    
    filename = file_path.split("/")[-1].split(".")[0]
    
    image = cv2.imread(file_path)
    image = iu.convert_image_to_bw(image, 120)
    image = cv2.resize(image, target_size)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output_path + filename + ".png", image)
    index += 1
    if index > 10 and dry_run:
        break
