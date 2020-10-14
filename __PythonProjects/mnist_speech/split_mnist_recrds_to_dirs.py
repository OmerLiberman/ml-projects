"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

split_mnist_recrds_to_dirs.py
----------------
This file is used for the mnist_recordings dataset.
The dataset includes all the files in single directory.
This script splits the audio files to separated directories.
==============================================================================
"""
import os
import shutil

curr_path = "/Users/omerliberman/Desktop/datasets/mnist_recordings/"
new_path = "/Users/omerliberman/Desktop/datasets/mnist_recordings/mnist_recordings_ds/"

classes = [num for num in range(0, 10)]

"""
All the next block should be ignored if the directories already exists.
"""
# Create the top directory.
os.mkdir(new_path)
# Create the all sub directories.
for cls in classes:
	new_dir_path = new_path + str(cls)
	os.mkdir(new_dir_path)


for file in os.listdir(curr_path):
	if file.endswith('.wav'):
		label = file[0]  # all files starts with the label.
		print(file)
		src = curr_path + file
		dst = new_path + label + "/" + file
		shutil.copy(src=src, dst=dst)
