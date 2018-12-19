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
	# and store all the latter four columns in a new column, called 'text_message' (5th column)
	data.fillna(' ', inplace = True)
	data['text_message'] = data['v2'] + data['Unnamed: 2'] + data['Unnamed: 3'] + data['Unnamed: 4']
 	# the value stored in the 'text_message' was striped, and stored in lowercase.
	data['text_message'] = data['text_message'].str.strip()
	data['text_message'] = data['text_message'].str.lower()
	# I used the LabelEncoder method to transform the spam and ham to 1 and 0, respectively
	# It stores in the 6th column of the data
	le = preprocessing.LabelEncoder()
	data['label_enc'] = le.fit_transform(data['v1'])
	# In order to avoid the influence of data's input order, the entire data is shuffled through data.sample(frac=1)method
	# reset_index(drop=True) is used to avoid an extra column for old index.
	data = data.sample(frac=1).reset_index(drop = True)
	# generate a str of punctuations, and translate the text with no punctuations
	punc = string.punctuation
	table = str.maketrans('', '', punc) 
	data['text_message'] = data['text_message'].apply(lambda x: x.translate(table))
	# the 7th column split the sent by whitespace
	data['word_token'] = data.apply(lambda x:x['text_message'].split(' '), axis = 1)
	# the 8th column in data stores stopwords removed SMS text msgs
	stop_words = set(stopwords.words('english'))
	data['cleaned_text'] = data.apply(lambda x: [word for word in x['word_token'] if word not in stop_words], axis = 1)
	# the 9th clumn in data stores the words stemmed by porter stemmer 
	stemmer = PorterStemmer()
	data['stemmed'] = data.apply(lambda x: [stemmer.stem(word) for word in x['cleaned_text']], axis = 1)
	# the 10th column in data includes the final SMS text msgs, which removes the uncommonly used one letter word 
	data['final_text'] = data.apply(lambda x: ' '.join([word for word in x['stemmed'] if len(word) > 1]), axis = 1)
	return data

## this method returns a dataframe with feature sets, which are tfidf coefs
def extract_tfidf_feature_sets(data):
	# create an empty dictionary
	vocab = dict()
	# the keys in the vocab dictionary are the tokens appeared in the entire data, 
	# and their corresponding values are how many times they appeared in the data
	for text in data['final_text'].values:
		for ele in text.split(" "):
			vocab[ele] = vocab.get(ele, 0) + 1
	df = dict()
	for word in vocab.keys():
		df[word] = np.sum(data['final_text'].apply(lambda x: 1 if word in x else 0))
	# idf 1 + log(N/(1+ d(w))) in case d(w) = 0
	idf = {k:1+np.log(data.shape[0]/(1+v)) for k, v in df.items()}
	for ele in vocab.keys():
		data[ele] = data['final_text'].apply(lambda x: x.count(ele)* idf[ele] if ele in x else 0)
	return data

## this method returns a dataframe with feature sets, which are the counts of words appeared in a single documents.
def extact_count_feature_sets(data):
	vocab = dict()
	for text in data['final_text'].values:
		for ele in text.split(" "):
			vocab[ele] = vocab.get(ele, 0) + 1
	for ele in vocab.keys():
		data[ele] = data['final_text'].apply(lambda x: x.count(ele) if ele in x else 0)
	return data

## this method returns a dataframe with feature sets, which are the words percentages in each document.
def extact_percentage_feature_sets(data):
	vocab = dict()
	for text in data['final_text'].values:
		for ele in text.split(" "):
			vocab[ele] = vocab.get(ele, 0) + 1
	for ele in vocab.keys():
		data[ele] = data['final_text'].apply(lambda x: x.count(ele)/float(len(x) + 1) if ele in x else 0)
	return data

def test_accuracy(data):
	X = data.iloc[:, 11:]
	y = data.iloc[:, 6]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
	model = LogisticRegression()
	model.fit(X_train, y_train)
	score = model.score(X_test, y_test)
	print(score)
	return score

if __name__ == '__main__':
	## use pandas to read data into DataFrame
	data = pd.read_csv('spam.csv',encoding = "ISO-8859-1")[:1000]
	data = preprocessing_data(data)
	data_with_tfidf_feature_sets = extract_tfidf_feature_sets(data)
	data_with_count_feature_sets = extact_count_feature_sets(data)
	data_with_percentage_feature_sets = extact_percentage_feature_sets(data)
	accuracy_tdidf = test_accuracy(data_with_tfidf_feature_sets)
	accuracy_count = test_accuracy(data_with_count_feature_sets)
	accuracy_percentage = test_accuracy(data_with_percentage_feature_sets)



