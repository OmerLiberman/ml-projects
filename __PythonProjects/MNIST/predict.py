"""
predict.py
----------------
Load trained model and predict inputs.
"""

import argparse
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import asarray

parser = argparse.ArgumentParser(description='Predict - MNIST')
parser.add_argument('--predict', help='full path to image should be predicted.')
args = parser.parse_args()

img_to_predict = args.predict

model = keras.models.load_model('mnist_model.h5')


# load an image and predict the class
def predict(img_path):
	"""
	:param img_path: Path to SINGLE img!
	:param show_img: boolean. show the img or not.
	:return: prediction (string of digit)
	"""
	# reshape into a single sample with 1 channel
	X_test = []
	img = load_img(img_path, target_size=(28, 28, 1), color_mode="grayscale")
	img = img_to_array(img)
	img = img.astype('float32')
	img = img / 255
	X_test.append(img)
	test = asarray(X_test)

	# predict the class
	prediction = model.predict_classes(test)
	predict_prob = model.predict_proba(test)[0]

	print("Prediction is: {}\nThe probability: {}".format(str(prediction[0]), str(predict_prob[prediction][0])))


predict(img_to_predict)