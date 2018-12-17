""" 
<1>use three feature extractors:
1 the raw Count with CountVectorizer from sklearn
2 TF-IDF with TfidfVectorizer from sklearn
3 binary feature: check if each of the most frequent 200 words is in the email with self difined function
<2>use Adaboost from sklearn as an example model to test the feature extractor and the evaluation methods
"""


import numpy as np
import string
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction import DictVectorizer
import random
import nltk
from nltk.corpus import senseval
from nltk.corpus.reader.senseval import SensevalInstance
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, sent_tokenize


#load the data as csv format
data = pd.read_csv('spam.csv',encoding = "ISO-8859-1")

#use count as feature
count_vectorizer = CountVectorizer(decode_error='ignore')
X = count_vectorizer.fit_transform(data['v2'])
Y = data['v1'].values
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)



#build Adaboost classifier
model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print("Classification rate for AdaBoost with count vector:", model.score(Xtest, Ytest))


# use self difined binary value as feature


#code from preprocessing.py to generate labeled data 
raw_text = list(data['v2'])
label = list(data['v1'])
labeled_data = list(zip(raw_text, label))


# steps to generate the most frequent 200 words within the give data base
#tokenize all the words from the data base
tokens =[word_tokenize(i) for(i,w) in labeled_data]
tokens_string1=[" ".join(i) for i in tokens]
tokens_string2=[" ".join(tokens_string1)]
tokens3 = [word for sent in sent_tokenize(str(tokens_string2)) for word in word_tokenize(sent)]


# filtering out puctuations
a=[w for w in tokens3 if w not in string.punctuation]


# filtering out stop words
filtered_words = [w for w in a if not w in stop_words]


# calculate the frequence of each words and get the 200 most frequent words
all_words = nltk.FreqDist(w.lower() for w in filtered_words)
word_features = [i for (i,w) in all_words.most_common(200)]

# function to extract features of contain each of the 200 most frequent words or not
def file_features(file): 
        file_words = set(file) 
        features = {}
        for word in word_features:
            features['contains({})'.format(word)] = (word in file_words)
        return features

#generate feature set 
featuresets = [(file_features(n), label) for (n, label) in labeled_data]

#vectorize feature set with DictVectorizer from sklearn
vec1 = DictVectorizer(sparse=False)
X = vec1.fit_transform([i for (i,w) in featuresets])


# get Y vector
Y = [w for (i,w) in featuresets]

#split X and Y into train set and test set
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)

#build adaboost classifier again with the binary feature
model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print("Classification rate for AdaBoost with binary feature:", model.score(Xtest, Ytest))



