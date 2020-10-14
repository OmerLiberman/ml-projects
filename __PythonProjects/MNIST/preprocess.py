"""
preprocess.py
----------------
Outputs 4 numpy arrays saved in .npy files.
"""

import argparse
from os import listdir
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import asarray, save
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='MNIST Preprocessing')
parser.add_argument('--dataset', dest='dataset', help="""Path to dataset which includes 10 different folders
													named by number 0 to 9""")
args = parser.parse_args()

DATASET_PATH = args.dataset

IMG_TYPE = ".jpg"
CLASSES = 10
IMG_SIZE = (28, 28)

X, y = [], []


# Helper Method - One Hot Encoding
def one_hot_encoding(arr, num_of_classes):
	one_hot_encoded = []
	for value in arr:
		letter = [0 for _ in range(num_of_classes)]
		letter[value] = 1
		one_hot_encoded.append(asarray(letter))
	return asarray(one_hot_encoded)


# Images loading
# The folders are named: "0", "1", ..., "9".
for folder in range(0, 10):
	folder_path = DATASET_PATH + str(folder) if DATASET_PATH.endswith("/") else DATASET_PATH + "/" + str(folder)
	for file in listdir(folder_path):
		if file.endswith(IMG_TYPE):
			img_path = folder_path + "/" + file
			label = folder

			# Img loading.
			img = load_img(img_path, target_size=IMG_SIZE, color_mode="grayscale")

			# Convert img to numpy array.
			img = img_to_array(img)

			X.append(img)
			y.append(label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = asarray(X_train)
X_train = X_train.astype('float32')
X_train /= 255

X_test = asarray(X_test)
X_test = X_test.astype('float32')
X_test /= 255

y_train = one_hot_encoding(y_train, CLASSES)
y_test = one_hot_encoding(y_test, CLASSES)

print("Train set sizes: X - {}, y - {}".format(X_train.shape, y_train.shape))
print("Test set sizes: X - {}, y - {}".format(X_test.shape, y_test.shape))

# Saving the images and the labels
save('X_train', X_train)
save('y_train', y_train)

save('X_test', X_test)
save('y_test', y_test)

print("Preprocessing ended.")