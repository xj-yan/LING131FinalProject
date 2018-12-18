from models.NeuralNetwork import NeuralNetwork
from sklearn import preprocessing
import numpy as np
import os
import pandas as pd
from nltk import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVetorizer

def feature_extraction_1(X):
    return [helper_1(text) for text in X]

def helper_1(text):
	tokens = word_tokenize(text)

    return 1

def feature_extraction_2(X):
    count_vectorizer = CountVectorizer(decode_error='ignore')
    X = count_vectorizer.fit_transform(X)
    return X
def feature_extraction_3(X, Y):
	tfidf_vectorizer = TfidfVetorizer
	X = tfidf_vectorizer.fit_transform(X)
    return X

def read_from_file(file_name):
    data = pd.read_csv(file_name, encoding = "ISO-8859-1")
    raw_text = list(data['v2'])
    label = list(data['v1'])
    return raw_text, label

def train_test_split(X, Y, percentage):
    length = len(X)
    split = int(length * percentage)
    X_train = X[0: split]
    y_train = Y[0: split]
    X_test = X[split + 1:]
    y_test = Y[split + 1:]
    return X_train, y_train, X_test, y_test

def main():
	# read from file
    X, Y = read_from_file('spam.csv')
    # extract feature 1
    X = feature_extraction_1(X)
    X = np.array(X)
    X = preprocessing.scale(X) # feature scaling
    Y = np.array(Y)
    X_train, y_train, X_test, y_test = train_test_split(X, Y, 0.33)

    print('Type 1 Feature:')
    # hidden layer: 5 neurons in hidden layer
    # classifier1: feature_extraction_1
    classifier1 = NeuralNetwork(X_train, y_train, 4)
    classifier1.train()
    print('Self-built NeuralNetwork Accuracy: ', classifier1.accuracy(X_test, y_test))

    print('Type 2 Feature:')
    # classifier2: feature_extraction_2
    classifier2 = NeuralNetwork(X_train, y_train, 4)
    classifier2.train()
    print('Self-built NeuralNetwork Accuracy: ', classifier2.accuracy(X_test, y_test))

    print('Type 3 Feature:')
    # classifier3: feature_extraction_3
    classifier3 = NeuralNetwork(X_train, y_train, 4)
    classifier3.train()
    print('Self-built NeuralNetwork Accuracy: ', classifier3.accuracy(X_test, y_test))

    print('Type 4 Feature:')
    # classifier4: feature_extraction_4
    classifier4 = NeuralNetwork(X_train, y_train, 4)
    classifier4.train()
    print('Self-built NeuralNetwork Accuracy: ', classifier4.accuracy(X_test, y_test))

if __name__ == '__main__':
    main()