import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
from scipy import signal
from tensorflow import keras
from scipy.io import wavfile
from keras.preprocessing.image import img_to_array


# --- Constants.
_spectrogram_size = (128, 128)
model = keras.models.load_model('model.h5')
with open('dict.json') as f:
	dict_of_labels = json.load(f)


# --- Helpers.
def _get_single_spectrogram_img(path_to_wav_file, input_shape=_spectrogram_size):
	"""
	Gets path to .wav file and returns his spectrogram as numpy array.
	"""
	sample_rate, samples = wavfile.read(path_to_wav_file)
	frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
	img = Image.fromarray(spectrogram, 'RGB')
	img.save('curr_img.jpg')
	image = tf.keras.preprocessing.image.load_img('curr_img.jpg', target_size=input_shape)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image /= 255.
	os.remove('curr_img.jpg')
	return image


# --- Globals.
def predict(path_to_wav_file):
	spectrogram = _get_single_spectrogram_img(path_to_wav_file)
	prediction = model.predict(spectrogram)[0]
	max_index = np.argmax(prediction)

	predicted_label = dict_of_labels[str(max_index)]

	print("The prediction is: {}\nProbability: {}".format(predicted_label, prediction[max_index]))
	return predicted_label


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


# --- Testing.

# path1 = "/Users/omerliberman/Desktop/datasets/voice/smaller_ds/"
# all_sub_dirs = _get_all_sub_dir(path1)
# tot, pos = 0., 0.
# for subdir in all_sub_dirs:
# 	label = _extract_label(subdir)
# 	for file in os.listdir(subdir):
# 		if file.endswith('.wav'):
# 			full_path = subdir + "/" + file
# 			predicted_label = predict(full_path)
# 			if predicted_label == label:
# 				pos += 1.
# 			tot += 1.
#
# print(pos/tot)
