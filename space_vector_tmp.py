import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
	data = pd.read_csv('spam.csv',encoding = "ISO-8859-1")
	data.fillna(' ', inplace = True)
	data['text_message'] = data['v2'] + data['Unnamed: 2'] + data['Unnamed: 3'] + data['Unnamed: 4']
	data['text_message'] = data['text_message'].str.strip()
	data['text_message'] = data['text_message'].str.lower()
	le = preprocessing.LabelEncoder()
	data['label_enc'] = le.fit_transform(data['v1'])
	data = data.sample(frac=1).reset_index(drop = True)
	# data.head()
	# data.info()
	# for i in range(5):
	# 	print(data['text_message'][i])
	# 	print(data['label_enc'][i])
	punc = string.punctuation
	table = str.maketrans('', '', punc)
	stop_words = set(stopwords.words('english'))
	data['text_message'] = data['text_message'].apply(lambda x: x.translate(table))
	data['word_token'] = data.apply(lambda x:x['text_message'].split(' '), axis = 1)
	data['cleaned_text'] = data.apply(lambda x: [word for word in x['word_token'] if word not in stop_words], axis = 1)
	stemmer = PorterStemmer()
	data['stemmed'] = data.apply(lambda x: [stemmer.stem(word) for word in x['cleaned_text']], axis = 1)
	data['final_text'] = data.apply(lambda x: ' '.join([word for word in x['stemmed'] if len(word) > 1]), axis = 1)
	# data.info()
	# <class 'pandas.core.frame.DataFrame'>
	# RangeIndex: 5572 entries, 0 to 5571
	# Data columns (total 10 columns):
	# v1              5572 non-null object
	# v2              5572 non-null object
	# Unnamed: 2      5572 non-null object
	# Unnamed: 3      5572 non-null object
	# Unnamed: 4      5572 non-null object
	# text_message    5572 non-null object
	# word_token      5572 non-null object
	# cleaned_text    5572 non-null object
	# stemmed         5572 non-null object
	# final_text      5572 non-null object
	# dtypes: object(10)
	# memory usage: 435.4+ KB
	vocab = dict()
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
	# size = len(data)
	# training_size = round(0.75 * size)
	# X_train = data.iloc[:training_size, 10:].values()
	# X_test = data.iloc[training_size:, 10:].values()
	# Y_train = data.iloc[:training_size, 6].values()
	# Y_test = data.iloc[training_size:,6].values()
	X = data.iloc[:, 11:]
	y = data.iloc[:, 6]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
	model = LogisticRegression()
	model.fit(X_train, y_train)
	print(model.score(X_test, y_test))


