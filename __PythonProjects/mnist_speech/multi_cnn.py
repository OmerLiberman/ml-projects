"""
multi_cnn.py
---------
"""
import os
import json
import argparse
import tensorflow as tf

from numpy import append, asarray
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

import warnings

from tensorflow.python.keras.layers import Dropout

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def _init_model(num_of_classes, img_width, img_height, conv_wind_width, conv_wind_height, pool_wind_width, pool_wind_height, actv_fnc_hidn_layers, actv_fnc_outpt_layers):
	"""
	img_width, img_height - sizes of the input image.
	conv_wind_width, conv_wind_height - sizes of the convolution window.
	pool_wind_width, pool_wind_height - sizes of the pooling window.
	activation_func_hidden_layers, activation_func_output_layer are the types of the activation funcs.
	"""
	conv_wind = (conv_wind_width, conv_wind_height)
	pool_wind = (pool_wind_width, pool_wind_height)

	model = tf.keras.models.Sequential()

	model.add(tf.keras.layers.Conv2D(32, kernel_size=conv_wind, input_shape=(img_width, img_height, 3), activation=actv_fnc_hidn_layers))
	model.add(tf.keras.layers.Conv2D(64, kernel_size=conv_wind, activation=actv_fnc_hidn_layers))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_wind))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(128, activation=actv_fnc_hidn_layers))
	model.add(Dropout(0.5))
	model.add(tf.keras.layers.Dense(num_of_classes, activation=actv_fnc_outpt_layers))

	model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
	print(model.summary())
	return model


def _cast_types(args):
	"""
	Cast the types from strings to their types.
	"""
	# epochs.
	args.epochs = int(args.epochs)

	# batch.
	args.batch = int(args.batch)

	# img_width.
	args.img_width = int(args.img_width)

	# img_height.
	args.img_height = int(args.img_height)

	# conv_width.
	args.conv_width = int(args.conv_width)

	# conv_height.
	args.conv_height = int(args.conv_height)

	# pool_width.
	args.pool_width = int(args.pool_width)

	# pool_height.
	args.pool_height = int(args.pool_height)

	return args


def _load_images_from_path(path, input_type, input_shape, label):
	"""
	Receives a path to folder and return X, y where both are arrays.
	X is an array of numpy arrays.
	"""
	path = path if path.endswith("/") else path + "/"
	X, y = [], []
	for file in os.listdir(path):
		if file.endswith(input_type):
			image_full_path = path + file

			# Image loading
			image = tf.keras.preprocessing.image.load_img(image_full_path, target_size=input_shape)

			# Convert image to numpy array
			image = img_to_array(image)

			# Store
			X.append(image)
			y.append(label)
	X, y = asarray(X), asarray(y)
	return X, y


def _get_all_sub_folders(folder):
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


def _create_dict_of_labels(labels):
	# Creating dictionary for the labels.
	numerical_labels = [i for i in range(len(labels))]
	dict_of_labels = zip(labels, numerical_labels)
	with open('labels_dictionary.json', 'w') as fp:
		json.dump(dict(dict_of_labels), fp)
	fp.close()
	return numerical_labels


def main(args):
	"""
	main func.
	"""
	args = _cast_types(args)

	# Loading data to numpy arrays.
	paths_to_classes = args.dir
	all_sub_folders = _get_all_sub_folders(paths_to_classes)
	labels = [_extract_label(path) for path in all_sub_folders]
	labels_in_numbers = _create_dict_of_labels(labels)

	input_type = args.input_type
	input_shape = (args.img_width, args.img_height)

	X, y = None, None
	NUM_OF_CLASSES = len(all_sub_folders)

	for label in range(NUM_OF_CLASSES):
		X_curr, y_curr = _load_images_from_path(path=all_sub_folders[label], input_type=input_type, input_shape=input_shape,
												label=labels_in_numbers[label])

		if X is None:
			X, y = X_curr, y_curr
		else:
			X = append(X, X_curr, axis=0)
			y = append(y, y_curr)

	X /= 255.

	print(y)
	y = to_categorical(y)
	print(y)

	# Split to train/test.
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

	# model initiation.
	model = _init_model(
		len(all_sub_folders),
		args.img_width,
		args.img_height,
		args.conv_width,
		args.conv_height,
		args.pool_width,
		args.pool_height,
		args.hidden_activation,
		args.output_activation
	)

	BATCH_SIZE, EPOCHS = args.batch, args.epochs

	# train.
	hist = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))
	loss_train, acc_train = hist.history['loss'], hist.history['acc']
	print('cnvrg_tag_loss_train:', loss_train)
	print('cnvrg_tag_accuracy_train:', acc_train)

	# test.
	score = model.evaluate(X_test, y_test)
	loss_test, acc_test = score[0], score[1]

	print('cnvrg_tag_loss_val:', loss_test)
	print('cnvrg_tag_accuracy_val:', acc_test)

	# Save.
	where_to_save = args.project_dir + "/" + args.model if args.project_dir is not None else args.model
	model.save(where_to_save)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="""Deep Learning CNN Model for binary classification (images)""")

	parser.add_argument('--data', action='store', dest='dir', required=True, help="""Array of strings. path to the folders of the images of all classes. """)

	parser.add_argument('--input_type', action='store', dest='input_type', default=".jpg", help="The type of the images.")

	parser.add_argument('--project_dir', action='store', dest='project_dir', help="""String.. """)

	parser.add_argument('--output_dir', action='store', dest='output_dir', help="""String.. """)

	parser.add_argument('--model', action='store', default="cnn_model.h5", dest='model', help="""String. The name of the output file which is a trained model """)

	parser.add_argument('--epochs', action='store', default="20", dest='epochs', help="""Num of epochs. Default is 20""")

	parser.add_argument('--batch', action='store', default="128", dest='batch', help="""batch size. Default is 128""")

	parser.add_argument('--img_width', action='store', default="200", dest='img_width', help=""" The width of the input images .Default is 200""")

	parser.add_argument('--img_height', action='store', default="200", dest='img_height', help=""" The height of the input images .Default is 200""")

	parser.add_argument('--conv_width', action='store', default="3", dest='conv_width', help=""" The width of the convolution window.Default is 3""")

	parser.add_argument('--conv_height', action='store', default="3", dest='conv_height', help=""" The height of the convolution window.Default is 3""")

	parser.add_argument('--pool_width', action='store', default="2", dest='pool_width', help=""" The width of the pooling window.Default is 3""")

	parser.add_argument('--pool_height', action='store', default="2", dest='pool_height', help=""" The height of the pooling window.Default is 3""")

	parser.add_argument('--hidden_activation', action='store', default='relu', dest='hidden_activation', help="""The activation function for the hidden layers.""")

	parser.add_argument('--output_activation', action='store', default='softmax', dest='output_activation', help="""The activation function for the output layer.""")

	args = parser.parse_args()

	main(args)
