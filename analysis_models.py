from models.NeuralNetwork import NeuralNetwork
from models.NaiveBayes import NaiveBayes
from sklearn import preprocessing
import numpy as np
import copy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# feature extraction based on analysis_sklearn.py

# 1. raw count
def raw_count_feature(X):
    count_vectorizer = CountVectorizer(decode_error='ignore')
    X = count_vectorizer.fit_transform(X)
    return X.toarray()

# 2. tfidf
def tfidf_feature(X):
    tfidf_vectorizer = TfidfVectorizer(decode_error='ignore')
    X = tfidf_vectorizer.fit_transform(X)
    return X.toarray()

# run NaiveBayes classifier
def run_naivebayes_classifier(x_train, y_train, x_test, y_test):
    nb_classifier = NaiveBayes(copy.copy(x_train), copy.copy(y_train))
    nb_classifier.train()
    print('Self-built NaiveBayes Accuracy with Raw Count: ', nb_classifier.accuracy(copy.copy(x_test), copy.copy(y_test)))


# run NeuralNetwork classifier
def run_neuralnetwork_classifier(x_train, y_train, x_test, y_test, hidden_layer_size, iteration_count):
    nn_classifier = NeuralNetwork(copy.copy(x_train), copy.copy(y_train), hidden_layer_size)
    nn_classifier.train(iteration_count)
    print('Self-built NeuralNetwork Accuracy with Raw Count: ', nn_classifier.accuracy(copy.copy(x_test), copy.copy(y_test)))


# read raw data and label from file
def read_from_csv(file_name):
    data = pd.read_csv(file_name, encoding = "ISO-8859-1")
    raw_text = list(data['v2'])
    label = list(data['v1'])
    return raw_text, label

# split the training set and the test set by a percentage
def train_test_split(X, Y, percentage):
    length = X.shape[0]
    split = int(length * percentage)
    X_train = X[0: split]
    y_train = Y[0: split]
    X_test = X[split + 1:]
    y_test = Y[split + 1:]
    return X_train, y_train, X_test, y_test

def standardize_y(Y):
    y = []
    for word in Y:
        if word == 'spam':
            y.append([1])
        else:
            y.append([0])
    return np.array(y)

def main():
    # read from file
    X, Y = read_from_csv('spam.csv')
    # standardize y
    y = standardize_y(Y)
    # 1. Raw Count
    print('Feature: Raw Count')
    # Step 1: feature extraction
    x1 = raw_count_feature(copy.copy(X))
    # Step 2: feature scaling
    x1 = preprocessing.scale(x1)
    # Step 3: divide into training and test set
    x_train, y_train, x_test, y_test = train_test_split(x1, y, 0.9)
    # Step 4: run classifiers
    #     Neural Network: 20 neurons in hidden layer; 2000 iterations
    #     neurons in hidden layer        iteration time              RESULT
    #             4                        600                59.06642728904848
    #             4                        400                57.091561938958705
    run_neuralnetwork_classifier(x_train, y_train, x_test, y_test, 4, 600)
    #     Naive Bayes:
    #     Result: 87.07
    run_naivebayes_classifier(x_train, y_train, x_test, y_test)

    # 2. TF-IDF
    print('Feature: TF-IDF')
    # Step 1: feature extraction
    x2 = tfidf_feature(copy.deepcopy(X))
    # Step 2: feature scaling
    x2 = preprocessing.scale(x2)
    # Step 3: divide into training and test set
    x_train, y_train, x_test, y_test = train_test_split(x2, y, 0.9)

    # Step 4: run classifiers
    #     Neural Network:
    #     neurons in hidden layer        iteration time              RESULT
    #             4                         600                91.38240574506284
    #             4                         400                88.50987432675045
    run_neuralnetwork_classifier(x_train, y_train, x_test, y_test, 4, 600)
    #     Naive Bayes:
    #     Result: 87.07
    run_naivebayes_classifier(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()