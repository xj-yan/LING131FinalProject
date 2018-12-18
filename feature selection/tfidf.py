import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
	## use pandas to read data into DataFrame
	data = pd.read_csv('spam.csv',encoding = "ISO-8859-1")
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