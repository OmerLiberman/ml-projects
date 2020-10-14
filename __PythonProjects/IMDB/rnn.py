"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io

cnvrg.io - Projects Example

last update: Nov 07, 2019.
-------------
rnn.py
==============================================================================
"""
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from cnvrg import Experiment
from sklearn.model_selection import train_test_split


def cast_types(args):
	# epochs.
	args.epochs = int(args.epochs)

	# batch_size.
	args.batch_size = int(args.batch_size)

	# input_shape.
	args.input_shape = args.input_shape.split(' ')
	for num in args.input_shape:
		if num != '':
			num = int(num)
	args.input_shape = tuple(args.input_shape)

	# ----- #
	return args


def init_model(input_shape):

	model = keras.Sequential()
	model.add(keras.layers.Embedding(input_shape[0], 16))
	model.add(keras.layers.GlobalAveragePooling1D())
	model.add(keras.layers.Dense(16, activation=tf.nn.relu))
	model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model


def main(args):
	args = cast_types(args)

	df = pd.read_csv(args.data)
	X, y = df.iloc[:, :-1], df.iloc[:, -1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	model = init_model(X.shape)   # <--- Doesn't work with the shape.

	train_metrics = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2)

	test_metrics = model.evaluate(X_test, y_test)

	# train_loss = list(np.round(train_metrics.history['loss'], 3))
	# train_acc = list(np.round(train_metrics.history['accuracy'], 3))
	# val_loss = list(np.round(train_metrics.history['val_loss'], 3))
	# val_acc = list(np.round(train_metrics.history['val_accuracy'], 3))
	test_loss = float(test_metrics[0])
	test_acc = float(test_metrics[1])

	exp = Experiment()
	exp.log_param("test_loss", test_loss)
	exp.log_param("test_acc", test_acc)

	model.save("model.h5")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="""RNN Classifier""")

	parser.add_argument('--data', action='store', dest='data', required=True, help="""String. path to csv file: The data set for the classifier. Assumes the last column includes the labels. """)

	parser.add_argument('--project_dir', action='store', dest='project_dir', help="""String.""")

	parser.add_argument('--output_dir', action='store', dest='output_dir', help="""String.""")

	parser.add_argument('--input_shape', action='store', dest='input_shape', default="10000", help="""The shape of the input. Look like: a b c.""")

	parser.add_argument('--epochs', action='store', default="10", dest='epochs', help="Number of epochs when training.")

	parser.add_argument('--batch_size', action='store', default="64", dest='batch_size', help="batch size when training.")

	args = parser.parse_args()

	main(args)

