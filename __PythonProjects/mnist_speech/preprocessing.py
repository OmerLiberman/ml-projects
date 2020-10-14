"""
processing.py
-------------
This file receives a path to a directory where each sub-directory includes .wav files.
This files mirrors (create equivalent copy) of the given directory, including sub directories and files.
The only difference is that each file is a spectrogram of a wav file, not the wav file itself.
"""
import os
import json
import argparse
import numpy as np

import librosa
from librosa import display
from PIL import Image
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

# --- Arguments.
parser = argparse.ArgumentParser(description="""Pre-processing for speech recognition.""")

parser.add_argument('--data', action='store', required=True, dest='data',
					help="""String. Path to directory with some sub-directories, each includes audio files.""")

parser.add_argument('--input_type', action='store', dest='input_type', default='.wav',
					help="""String. Path to directory with some sub-directories, each includes audio files.""")

parser.add_argument('--output_type', action='store', dest='output_type', default='.jpg',
					help="""String. Path to directory with some sub-directories, each includes audio files.""")

parser.add_argument('--output_dir_name', action='store', dest='output_dir_name', default='spectrograms',
					help="""String. Path to directory with some sub-directories, each includes audio files.""")

args = parser.parse_args()

# --- Constants.
_wav_files_top_dir = args.data

_input_file_type = args.input_type
_output_file_type = args.output_type

_spectrogram_size = (34, 50)

_top_dir_name = args.output_dir_name


# --- Helpers.
def _create_dir(dir_name):
	original_dir_name = dir_name
	file_created, count = False, 1
	while True:
		if not os.path.exists(dir_name):
			os.mkdir(dir_name)
			return dir_name
		else:
			dir_name = original_dir_name + str(count)
		count += 1


def _get_labels_dict(labels):
	"""
	Receives list of labels and return dict matches number to each label.
	"""
	to_return = dict()
	for i in range(len(labels)):
		to_return[i] = labels[i]
	return to_return


def _get_single_spectrogram_img(wav_file_path, img_file_path):
	y, sr = librosa.load(wav_file_path)
	librosa.display.waveplot(y, sr=sr)
	plt.axis('off')
	plt.savefig(img_file_path, bbox_inches='tight', dpi=100)
	plt.clf()


def _get_all_sub_dir(folder):
	"""
	Returns all the sub directories in the given folder.
	:param folder: Path to folder.
	:return: Array of strings. Each is a path to sub-dir.
	"""
	sub_folders = [f.path for f in os.scandir(folder) if f.is_dir() and not f.path.endswith(".cnvrg")]
	return sub_folders


def _extract_label(path):
	"""
	Extracts the label of the object from the path.
	"""
	split_path = path.split('/')
	split_path = [str for str in split_path if str != '']
	return split_path[-1]


def _mirror_files_of_dir(src, dst, src_files_type=_input_file_type, dst_files_type=_output_file_type):
	"""
	:param src: The source directory. includes wav files.
	:param dst: The output directory. Where the spectrogram is going to be.
	:param src_files_type: Type of the input file.
	:param dst_files_type: Type of the output file.
	:return:
	"""
	for file in os.listdir(src):
		if file.endswith(src_files_type):
			full_input_path = src + file if src.endswith('/') else src + '/' + file
			full_output_path = dst + file.replace(src_files_type, dst_files_type) if dst.endswith('/') else dst + '/' + \
																											file.replace(src_files_type,
																														 dst_files_type)
			_get_single_spectrogram_img(full_input_path, full_output_path)
	return


def _create_all_dirs(all_sub_dirs):
	"""
	Creating equivalent spectrogram for each wav file.
	:param all_sub_dirs:
	:return:
	"""
	for subdir in all_sub_dirs:
		eq_dir_path = _top_dir_name + "/" + _extract_label(subdir)
		print("Dir Created: ", eq_dir_path)
		_mirror_files_of_dir(subdir, eq_dir_path)
	return


def _create_dict_of_labels(all_sub_dirs):
	# Creating dictionary for the labels.
	labels_dict = _get_labels_dict([_extract_label(subdir) for subdir in all_sub_dirs])
	with open('labels_dict.json', 'w') as fp:
		json.dump(labels_dict, fp)
	fp.close()


def _scale_all_images():
	all_sub_dirs = _get_all_sub_dir("spectrograms")
	for sub_dir in all_sub_dirs:
		for img in [file for file in os.listdir(sub_dir) if file.endswith('.jpg')]:
			full_path = sub_dir + "/" + img
			img_obj = Image.open(full_path)
			img_obj = img_obj.resize(_spectrogram_size)
			img_obj = img_obj.convert('1')
			os.remove(full_path)
			img_obj.save(full_path)
	return


# ---- Main.

# Creating the top dir.
_top_dir_name = _create_dir(_top_dir_name)

# Get all sub-dirs of the original directory.
all_sub_dirs = _get_all_sub_dir(_wav_files_top_dir)

# Creating equivalent sub-dir in the new directory.
for subdir in all_sub_dirs:
	subdir_name = _extract_label(subdir)
	_create_dir(_top_dir_name + "/" + subdir_name)

# Create the spectrogram.
_create_all_dirs(all_sub_dirs)

# Creating dictionary for the labels.
_create_dict_of_labels(all_sub_dirs)

_scale_all_images()
