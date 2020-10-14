"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

predict.py
==============================================================================
"""
import pickle
import numpy as np

from cnvrg import Endpoint


def predict(*args):
	"""
	:param args: should get 4 floats.
	:return: prediction (string).
	"""
	loaded_model = pickle.load(open("kNearestNeighborsModel.sav", 'rb'))

	sepal_length, sepal_width, petal_length, petal_width = float(args[0]), float(args[1]), float(args[2]), float(args[3])
	print("Got - sepal_length: {}, sepal_width: {}, petal_length: {}, petal_width: {}".format(sepal_length, sepal_width, petal_length, petal_width))

	arr = np.array([sepal_length, sepal_width, petal_length, petal_width])
	prediction = loaded_model.predict(arr.reshape(1, -1))

	iris_dict = {'Iris-setosa': -1, 'Iris-versicolor': 0, 'Iris-virginica': 1}
	prediction = [k for k, v in iris_dict.items() if v == prediction][0]

	e = Endpoint()
	e.log_metric("Prediction", prediction)
	return prediction
