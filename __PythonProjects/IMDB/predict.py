import json
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence

# Loading the dictionary of words and their numeric value.
with open('words_and_values.json') as f:
	data = json.load(f)

# Constants.
punctuations = data['***punctuations***']
max_len = data['***MAX_LEN***']
padding = data['***padding***']
start_of_rev = data['***start_of_rev***']
unknown = data['***unknown***']

# Loading the model.
new_model = keras.models.load_model('model.h5')


def _process(review):
	"""
	Processes the given review.
	:param review: string.
	:return: array of numbers.
	"""
	processed_review = []
	words_in_review = review.split(' ')
	for word in words_in_review:
		# Remove punctuations.
		for ch in punctuations:
			word = word.replace(ch, ' ')
		# Remove redundant spaces (replaces 2 spaces by one).
		word = word.replace('  ', ' ')
		# Change all words to lower case.
		word = word.lower()
		# Get the numeric value of the word.
		numeric_val = data[word] if word in data.keys() else unknown
		processed_review.append(numeric_val)
	processed_review = processed_review[:max_len]
	processed_review = sequence.pad_sequences([processed_review], maxlen=max_len, value=padding)
	return processed_review


def predict(review):
	processed_rev = _process(review)
	prediction = new_model.predict(processed_rev)
	sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'
	result = "Sentiment is: {} \nScore is: {}".format(sentiment, prediction)
	return result
