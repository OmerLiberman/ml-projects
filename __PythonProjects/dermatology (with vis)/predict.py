"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io

cnvrg.io - Projects Example

last update: Nov 14, 2019.
-------------
predict.py
==============================================================================
"""
import pandas as pd
from sklearn.externals import joblib

loaded_model = joblib.load("model.sav")

columns = ['erythema', 'scaling', 'definite_borders', 'itching',
       'koebner_phenomenon', 'polygonal_papules', 'follicular_papules',
       'oral_mucosal_involvement', 'knee_and_elbow_involvement',
       'scalp_involvement', 'family_history', 'melanin_incontinence',
       'eosinophils_in_the_infiltrate', 'pnl_infiltrate',
       'fibrosis_of_the_papillary_dermis', 'exocytosis', 'acanthosis',
       'hyperkeratosis', 'parakeratosis', 'clubbing_of_the_rete_ridges',
       'elongation_of_the_rete_ridges',
       'thinning_of_the_suprapapillary_epidermis', 'spongiform_pustule',
       'munro_microabcess', 'focal_hypergranulosis',
       'disappearance_of_the_granular_layer',
       'vacuolisation_and_damage_of_basal_layer', 'spongiosis',
       'saw-tooth_appearance_of_retes', 'follicular_horn_plug',
       'perifollicular_parakeratosis', 'inflammatory_monoluclear_inflitrate',
       'band-like_infiltrate', 'age']

def _process(args):
	args = args[0].split(' ')
	to_return = []
	for arg in args:
		to_return.append(int(arg))
	if len(to_return) < 34:
		for i in range(len(to_return), 34):
			to_return.append(1)
	return to_return


def predict(*args):
	"""
	:param args: should get 34 floats, if its given less it completes it with 0's.
	:return: prediction (string) which is number in [1, 6]
	"""
	to_predict = _process(args)

	to_predict = pd.DataFrame(to_predict).transpose()
	to_predict.columns = columns
	prediction = loaded_model.predict(to_predict)

	return prediction[0]
