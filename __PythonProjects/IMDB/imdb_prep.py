"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io

cnvrg.io - Projects Example

last update: Nov 07, 2019.
-------------
imdb_prep.py
==============================================================================
"""
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

"""
Inputs from the user.
"""
parser = argparse.ArgumentParser(description="""Parser for the imdb preprocessor""")

parser.add_argument('--data', action='store', dest='data', required=True, help="""string. path to the imdb dataset.""")

parser.add_argument('--num_words', action='store', dest='num_words', type=int, default=1000, help="""max number of words to include. Words are ranked by how often they occur 
					(in the training set) and only the most frequent 
					words are kept""")

parser.add_argument('--skip_top', action='store', dest='skip_top', type=int, default=0, help="""skip the top N most frequently occurring words (which may not be informative).""")

parser.add_argument('--maxlen', action='store', dest='maxlen', type=int, default=500, help="""sequences longer than this will be filtered out.""")

args = parser.parse_args()

args.num_words -=  4

"""
Parsing the csv file.
"""
df = pd.read_csv(args.data)
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})

# --- Constants.

# The number of words used in the reviews.
NUM_WORDS = args.num_words

# The number of words to skip from the top.
SKIP_TOP = args.skip_top

# The maximal length of review to accept (reviews longer than are sliced).
MAX_LEN = args.maxlen

# I set: 0 -> padding, 1 -> start, 2 -> unknown.
PADDING, START_OF_REVIEW, UNKNOWN, FIRST_VAL_OF_WORDS = 0, 1, 2, 3


# --- Helpers.
punctuations = ["<br />", '.', ',', '-', '"', '(', ')', '!', ':', '\'', ';', '/', '?', '*']
def _edit_textual_reviews(reviews):
	"""
	Process reviews - removing punctuations and redundant spaces.
	"""
	processed_reviews = []
	for rev_num in range(df.shape[0]):
		rev = df.review[rev_num]

		# Remove punctuations.
		for ch in punctuations:
			rev = rev.replace(ch, ' ')

		# Remove redundant spaces (replaces 2 spaces by one).
		rev = rev.replace('  ', ' ')

		# Change all words to lower case.
		rev = rev.lower()

		processed_reviews.append(rev)
	return processed_reviews


def _count_frequencies_of_words(reviews):
	"""
	Count frequencies of words in all reviews.
	"""
	dict_of_freq = dict()
	for rev_num in range(df.shape[0]):
		curr_rev = df.review[rev_num]
		words_in_rev = curr_rev.split(' ')

		for word in words_in_rev:
			if word in dict_of_freq.keys():
				dict_of_freq[word] += 1
			else:
				dict_of_freq[word] = 1
	return dict_of_freq


def _create_dict_of_words_and_their_rank_in_frequencies(words_in_reviews_sorted_by_freq):
	"""
	Matching each word to his rank in the number of frequencies.
	I set: 0 -> padding, 1 -> start, 2 -> unknown.
	:param words_in_reviews_sorted_by_freq: array of words. In the i'th cell there is the word which was used
	the i'th-greatest number of times.
	:return: dictionary like {word: number_replacing_the_word}
	"""
	word_rank_dict = dict()  # <-- This dict is like {word: #appears_in_all_reviews}
	# Insert words.
	for rank in range(len(words_in_reviews_sorted_by_freq)):
		if SKIP_TOP is not None and rank < SKIP_TOP:
			word_rank_dict[words_in_reviews_sorted_by_freq[rank]] = UNKNOWN
		elif NUM_WORDS is not None and rank > NUM_WORDS:
			word_rank_dict[words_in_reviews_sorted_by_freq[rank]] = UNKNOWN
		else:  # <--- Rank is fine!
			word_rank_dict[words_in_reviews_sorted_by_freq[rank]] = rank + FIRST_VAL_OF_WORDS
	# Insert punctuations.
	word_rank_dict['***punctuations***'] = punctuations
	# Insert constants.
	word_rank_dict['***MAX_LEN***'] = MAX_LEN
	word_rank_dict['***padding***'] = PADDING
	word_rank_dict['***start_of_rev***'] = START_OF_REVIEW
	word_rank_dict['***unknown***'] = UNKNOWN
	return word_rank_dict


def _convert_all_reviews_to_numbers(arr_of_reviews, word_num_dict):
	"""
	This method receives all the reviews and a dictionary which directs what number each word should be
	converted to.
	:param arr_of_reviews: all reviews. (arr of strings)
	:param word_num_dict: dictionary like : {word: numeric_val}
	:return: two arrays: [reviews_converted_to_numbers] , [sentiments]
	"""
	reviews, sentiments = [], []
	for rev_num in range(len(df.review)):  # <--- Traveling all reviews.
		curr_rev = df.review[rev_num]  # <--- The current review.
		words_in_curr_rev = curr_rev.split(' ')  # <--- Array of strings includes the words of the review.
		# Checking if there are more than words than the user permits.
		if MAX_LEN is not None and len(words_in_curr_rev) > MAX_LEN:
			words_in_curr_rev = words_in_curr_rev[:MAX_LEN]
		nums_arr = [START_OF_REVIEW]
		for word in words_in_curr_rev:  # <--- Traveling each word in the sentence.
			val_of_wrd = dict_of_words_and_their_rank_in_frequencies[word]  # <--- Numeric val of word.
			nums_arr.append(val_of_wrd)
		# Adding reviews.
		reviews.append(nums_arr)
		# Adding sentiments.
		sentiments.append(df.sentiment[rev_num])
	return reviews, sentiments


# --- Processing.
df.review = _edit_textual_reviews(df.review)
dict_of_frequencies = _count_frequencies_of_words(df.review)
words_in_reviews_sorted_by_freq = sorted(dict_of_frequencies, key=dict_of_frequencies.get, reverse=True)
dict_of_words_and_their_rank_in_frequencies = _create_dict_of_words_and_their_rank_in_frequencies(words_in_reviews_sorted_by_freq)

# Create new csv where the reviews are converted to numbers.
# In this section we create the output .csv which includes reviews with numbers instead of words.
reviews, sentiments = _convert_all_reviews_to_numbers(df.review, dict_of_words_and_their_rank_in_frequencies)

max_len_of_review = MAX_LEN if MAX_LEN is not None else max([len(review) for review in reviews])

# Padding reviews (to have all in the same length).
reviews = tf.keras.preprocessing.sequence.pad_sequences(reviews, maxlen=MAX_LEN, value=PADDING)

processed_df = pd.DataFrame(reviews)
sentiments = np.asarray(sentiments).astype('float32')
processed_df['sentiment'] = sentiments

processed_df.to_csv("IMDB_processed.csv", index=False)

# --- Saving the dictionary.
with open('words_and_values.json', 'w') as fp:
	json.dump(dict_of_words_and_their_rank_in_frequencies, fp)
fp.close()
