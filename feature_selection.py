"""
The file is designed to generate feature sets.
I tried three ways to generate features sets, including tdidf coefficients, raw counts, and percentage of each word in single document.

Just use the command "python feature_selection.py" could run this script. 
The file would extract feature sets in three strategies and print their classification scores.

"""


import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

## preprocessing the source data, and store the processed data in the same data frame
def preprocessing_data(data):

	# Since some non-null object exists inunnamed: 2, unnamed: 3, and unnamed: 4, I replace null as an empty space, 
	# and store all the latter four columns in a new column, called 'content' (5th column)
	data.fillna(' ', inplace = True)
	data['content'] = data['v2'] + data['Unnamed: 2'] + data['Unnamed: 3'] + data['Unnamed: 4']
 	# the value stored in the 'content' was striped, and stored in lowercase.
	data['content'] = data['content'].str.strip()
	data['content'] = data['content'].str.lower()


	# I used the LabelEncoder method to transform the spam and ham to 1 and 0, respectively
	# It stores in the 6th column of the data
	le = preprocessing.LabelEncoder()
	data['label_enc'] = le.fit_transform(data['v1'])

	# In order to avoid the influence of data's input order, the corpus is shuffled through data.sample(frac=1)method
	# reset_index(drop=True) is used to avoid an extra column for old index.
	data = data.sample(frac=1).reset_index(drop = True)

	# generate a str of punctuations, and remove the punctuations in the text using str.maketrans()
	punc = string.punctuation
	translator = str.maketrans('', '', punc) 
	data['content'] = data['content'].apply(lambda x: x.translate(translator))

	# the 7th column stores the tokenized SMS text msgs.
	data['tokenized_words'] = data.apply(lambda x:x['content'].split(' '), axis = 1)

	# the 8th column in data stores stopwords removed SMS text msgs
	stop_words = set(stopwords.words('english'))
	data['stopwords_removed_text'] = data.apply(lambda x: [word for word in x['tokenized_words'] if word not in stop_words], axis = 1)
	
	# the 9th clumn in data stores the words stemmed by porter stemmer 
	stemmer = PorterStemmer()
	data['stemmed_words'] = data.apply(lambda x: [stemmer.stem(word) for word in x['stopwords_removed_text']], axis = 1)

	# the 10th column in data includes the final SMS text msgs, which removes the uncommonly used one letter word 
	data['to_be_analyzed'] = data.apply(lambda x: ' '.join([word for word in x['stemmed_words'] if len(word) > 1]), axis = 1)
	return data

## this method returns a dataframe with feature sets, which are tfidf coefs
def extract_tfidf_feature_sets(data):
	# create an empty dictionary
	token = dict()
	# the keys in the token dictionary are the tokens appeared in the corpus, 
	# and their corresponding values are how many times they appeared in corpus
	for text in data['to_be_analyzed'].values:
		for ele in text.split(" "):
			token[ele] = token.get(ele, 0) + 1
	
	# The df dictionary is to record how many times each word appears in the corpus
	df = dict()
	for word in token.keys():
		df[word] = np.sum(data['to_be_analyzed'].apply(lambda x: 1 if word in x else 0))
	
	# In order to calculate idf, I used the formula idf = 1 + log(N/(1+ d(w))) 
	# (N is the number of documents in corpus, while d(w) is the number of word w in document d)
	# I used log(n/(1+d(w))) instead of log(n/d(w) in case of d(w) = 0
	idf = {k:1+np.log(data.shape[0]/(1+v)) for k, v in df.items()}

	# store the tdidf into the dataframe
	for ele in token.keys():
		data[ele] = data['to_be_analyzed'].apply(lambda x: x.count(ele)/float(len(x)+1)* idf[ele] if ele in x else 0)
	return data

## this method returns a dataframe with feature sets, which are the counts of words appeared in a single documents.
def extact_count_feature_sets(data):
	# create an empty dictionary, and calculated how many time they appeared in corpus
	token = dict()
	for text in data['to_be_analyzed'].values:
		for ele in text.split(" "):
			token[ele] = token.get(ele, 0) + 1

	# store the count information in dataframe
	for ele in token.keys():
		data[ele] = data['to_be_analyzed'].apply(lambda x: x.count(ele) if ele in x else 0)
	return data

## this method returns a dataframe with feature sets, which are the words percentages in each document.
def extact_percentage_feature_sets(data):
	# create an empty dictionary, and calculated how many time they appeared in corpus
	token = dict()
	for text in data['to_be_analyzed'].values:
		for ele in text.split(" "):
			token[ele] = token.get(ele, 0) + 1
	# store the percentages of each word in document d in dataframe
	for ele in token.keys():
		data[ele] = data['to_be_analyzed'].apply(lambda x: x.count(ele)/float(len(x) + 1) if ele in x else 0)
	return data

def test_accuracy(data):
	# slice the dataframe, the X is the input data from column 11 to the 
	# last column, and the y is the encoded labels, which is in the 6th column.
	X = data.iloc[:, 11:]
	y = data.iloc[:, 6]
	# split the train and test set with method train_test_split() from sklearn.model_selection
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
	model = LogisticRegression()
	model.fit(X_train, y_train)
	score = model.score(X_test, y_test)
	print(score)
	return score

if __name__ == '__main__':
	## use pandas to read data into DataFrame
	data = pd.read_csv('spam.csv',encoding = "ISO-8859-1")
	data = preprocessing_data(data)
	data_with_tfidf_feature_sets = extract_tfidf_feature_sets(data)
	data_with_count_feature_sets = extact_count_feature_sets(data)
	data_with_percentage_feature_sets = extact_percentage_feature_sets(data)
	print("Feature sets: count") 
	print("Accuracy scores: ")
	accuracy_count = test_accuracy(data_with_count_feature_sets)
	print("Feature sets: tf-idf")
	print("Accuracy scores: ")
	accuracy_percentage = test_accuracy(data_with_percentage_feature_sets)
	print("Feature sets: tf-idf")
	print("Accuracy scores: ")
	accuracy_tdidf = test_accuracy(data_with_tfidf_feature_sets)



