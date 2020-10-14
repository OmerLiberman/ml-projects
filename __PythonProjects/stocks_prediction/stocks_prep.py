"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

stocks_prep.py
==============================================================================
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser(description="Preprocessing for stocks dataset.")
parser.add_argument('--data', action='store', required=True, dest='data',
                    help='''Path to the csv file. The file should contain a column named 'Date' and
                    another 2 columns called 'Open' and 'Close' (in capital!). ''')
parser.add_argument('--target_col', action='store', dest='target', default='Close',
                    help='''The target column of the model.''')
parser.add_argument('--plot_graphs', action='store', dest='plot', default=False, type=bool,
                    help='''Boolean. Indicates whether to save the output graphs or not.''')
args = parser.parse_args()

data = pd.read_csv(args.data, index_col="Date", parse_dates=True)

def plot(data, feature):
	"""
	The method recevies a data frame and column name ('feature')
	and saves the graph's image to the current directory.
	"""
	plt.plot(data[feature])
	plt.xlabel("Year")
	plt.ylabel("Value")
	plt.title("{}".format(feature))
	plt.imsave("{}.jpg".format(feature))

new_data = pd.DataFrame()
new_data['Date'] = data.index
new_data.index = new_data['Date']
new_data = new_data.drop(['Date'], axis=1)
new_data[args.target] = data[args.target]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(new_data)

TEST_SET = 0.2
test_set_size = int(len(new_data) * TEST_SET)
train_set_size = len(new_data) - test_set_size

WINDOW = 60
X_train, y_train = [], []
for ind in range(WINDOW, train_set_size):
	X_train.append(scaled_data[ind - WINDOW:ind, 0])
	y_train.append(scaled_data[ind, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Fitting the shape to the neural network
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))

