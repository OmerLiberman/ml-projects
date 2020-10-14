"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

prep_iris.py
==============================================================================
"""
import argparse
import pandas as pd

# Receive the path to the csv table.
parser = argparse.ArgumentParser(description="""Preprocessor""")
parser.add_argument('--data', action='store', dest='data', required=True,
					help="""string. path to the raw dataset.""")
args = parser.parse_args()
data = args.data

# Open the table in DataFrame object.
data = pd.read_csv(data)

# Process the target column.
target_column = 'species'

# Remap the target column to numbers.
iris_dict = {'Iris-setosa': -1, 'Iris-versicolor': 0, 'Iris-virginica': 1}
data[target_column] = data[target_column].map(iris_dict)

data.to_csv("Iris_processed.csv")
