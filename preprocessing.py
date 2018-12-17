import re
import random
import math
import copy
import pandas as pd
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import numpy as np



def create_raw_data(data):
	# store the raw_text and label in seperate lists, and then zip them together
	# the first column is the raw_text, and the second column is the label, 'spam' or 'ham'
	raw_text = list(data['v2'])
	label = list(data['v1'])
	le = preprocessing.LabelEncoder()
	label_enc = le.fit_transform(label)
	raw_data = list(zip(raw_text, label_enc))[:30]
	random.shuffle(raw_data)
	return raw_data

def create_word_set(data):
	# loop over the entire data set, and store all the words in the 'v2' column in word set
	word_set = set()
	for item in data:
		tmp_set = set(str(item[0]).split(" "))
		word_set = word_set.union(tmp_set)
	return word_set

def compute_document_dict(word_set):
	# generate a dict, that keys are all words in the data, and values are set to zero
	document_dict = dict.fromkeys(word_set, 0)
	return document_dict

# def compute_tf(data, document_dict):
# 	# the data is the 'v2' column in raw_data(raw_data[,0])
# 	document_dict_tmp = copy.deepcopy(document_dict)
# 	tf_dict_list = []
# 	for i in range(len(data)):
# 		word_lst = str(data[i][0]).split(" ")
# 		document_tf_dict = {}
# 		length = len(word_lst)
# 		for word in word_lst:
# 			document_dict_tmp[word] += 1
# 		for word, count in document_dict_tmp.items():
# 			document_tf_dict[word] = count/float(length)
# 		tf_dict_list.append(document_tf_dict)
# 	return tf_dict_list

# def compute_tf(data, i, document_dict):
# 	# the data is the 'v2' column in raw_data(raw_data[,0])
# 	document_dict_tmp = copy.deepcopy(document_dict)
# 	word_lst = str(data[i][0]).split(" ")
# 	document_tf_dict = {}
# 	length = len(word_lst)
# 	for word in word_lst:
# 		document_dict_tmp[word] += 1
# 	for word, count in document_dict_tmp.items():
# 		document_tf_dict[word] = count/float(length)
# 	return document_tf_dict

def compute_tf(data, i):
	# the data is the 'v2' column in raw_data(raw_data[,0])
	word_lst = str(data[i][0]).split(" ")
	document_tf_dict = {}
	length = len(word_lst)
	for word in word_lst:
		document_tf_dict[word] = document_tf_dict.get(word, 0) + 1
	for word, count in document_tf_dict.items():
		document_tf_dict[word] = count/float(length)
	return document_tf_dict

# def compute_idf(data, document_dict):
# 	idf_dict = dict.fromkeys(document_dict.keys(), 0)
# 	length = len(data)
# 	for d_list in data:
# 		for word, count in d_list.items():
# 			if count > 0:
# 				idf_dict[word] += 1
# 	for word, count in idf_dict.items():
# 		idf_dict[word] = math.log2(length / float(count)) + 1
# 	return idf_dict

def compute_idf(entire_tf_dict_list, document_dict):
	idf_dict = dict.fromkeys(document_dict.keys(), 0)
	length = len(entire_tf_dict_list)
	for tf_dict in entire_tf_dict_list:
		for word, count in tf_dict.items():
			if count > 0:
				idf_dict[word] += 1
	for word, count in idf_dict.items():
		idf_dict[word] = math.log2(length / float(count)) + 1
	return idf_dict

def compute_tfidf(tf_dict, idf_dict):
	tfidf = dict()
	for word, value in tf_dict.items():
		tfidf[word] = value * idf_dict[word]
	return tfidf


def generate_features(i, tfidf_dict, document_dict):
	# use compute_tfidf to generate features, and sorted the dictionary by value, and limit the maximum number of features 5,000 
	features = dict.fromkeys(document_dict.keys(), 0)
	for word, value in tfidf_dict[i].items():
		features[word] = value
	return features

def create_feature_sets(raw_data, tfidf_dict, document_dict):
	# feature_sets = [(generate_features(i[0]), i[1]) for i in raw_data]
	feature_sets = []
	for i in range(len(raw_data)):
		tmp_tuple = (generate_features(i, tfidf_dict, document_dict), raw_data[i][1])
		feature_sets.append(tmp_tuple)
	size = len(feature_sets)
	training_size = round(0.9 * size)
	train_set, test_set = feature_sets[:training_size], feature_sets[training_size:]
	return train_set, test_set


if __name__ == '__main__':
	# read data from spam.csv
	data = pd.read_csv('spam.csv',encoding = "ISO-8859-1")
	# create raw data from the first two columns of original data 
	raw_data = create_raw_data(data)
	# generate word_set with all words in data
	word_set = create_word_set(raw_data)
	# generate a dict, that keys are all words in the data, and values are set to zero
	document_dict = compute_document_dict(word_set)

	entire_tf_dict_list = []
	for i in range(len(raw_data)):
		tmp_dict = compute_tf(raw_data, i)
		entire_tf_dict_list.append(tmp_dict)
	print(len(entire_tf_dict_list))
	idf_dict = compute_idf(entire_tf_dict_list, document_dict)
	tfidf_dict = []
	for i in range(len(raw_data)):
		tmp_tfidf = compute_tfidf(entire_tf_dict_list[i], idf_dict)
		tfidf_dict.append(tmp_tfidf)
	train_set, test_set = create_feature_sets(raw_data, tfidf_dict, document_dict)
	model = LogisticRegression()
	Xtrain_set = [d[0] for d in train_set]
	Ytrain_set = [d[1] for d in train_set]
	Xtest_set = [d[0] for d in test_set]
	Ytest_set = [d[1] for d in test_set]
	model.fit(Xtrain_set, Ytrain_set)
	print(model.score(Xtest_set, Ytest_set))

