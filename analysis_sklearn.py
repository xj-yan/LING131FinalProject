""" 
<1>use six feature extractors:
1 the raw Count with CountVectorizer from sklearn
2 TF-IDF with TfidfVectorizer from sklearn
3 binary feature: check if each of the most frequent 200 words is in the email with self difined function
4 bigram of words
5 ngram of characters within word bound
6 ngram of characters acorss word bound
<2>analyze with three model from sklearn
1 adaboost
2 MultinomialNB
3 MLPClassifier

"""


import numpy as np
import string
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction import DictVectorizer
import nltk
from nltk import word_tokenize, sent_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score


stop_words = set(stopwords.words('english'))


def load_data():
    #load the data as csv format
    data = pd.read_csv('spam.csv',encoding = "ISO-8859-1")
    return data


def count_features(data):
    #count the occurance of each word in a file
    count_vectorizer = CountVectorizer(decode_error='ignore')
    X = count_vectorizer.fit_transform(data['v2'])
    Y = data['v1'].values
    #Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    return X, Y

def TFIDF_features(data):
    #use TF-IDF to calculate feature
    tfidf = TfidfVectorizer(decode_error='ignore')
    X = tfidf.fit_transform(data['v2'])
    Y = data['v1'].values
    return X, Y


def Bigram_word_features(data):
    #calculate the bigram of words in each file
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1) 
    X = bigram_vectorizer.fit_transform(data['v2'])                              
    Y = data['v1'].values                              
    return X, Y

def ngram_char_features(data):
    #calculate the ngram of characters within a word bound, 
    #parameter n give the number of ngram. calculate exact that number of gram, 
    #means not including ngrams less than n.
    ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(5, 5))
    X=ngram_vectorizer.fit_transform(data['v2'])
    Y = data['v1'].values                              
    return X, Y


def ngram_char_crossW_features(data):
    #calculate ngram of characters, cross word bound is allowed
    ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(5, 5))
    X=ngram_vectorizer.fit_transform(data['v2'])
    Y = data['v1'].values                              
    return X, Y


# use self difined binary value as feature

def create_labeled_data(data):
    #code from preprocessing.py to generate labeled data 
    raw_text = list(data['v2'])
    label = list(data['v1'])
    labeled_data = list(zip(raw_text, label))
    return labeled_data

def tokenize_labeled_data(labeled_data):
    #tokenize all the words from the data base
    tokens =[word_tokenize(i) for(i,w) in labeled_data]
    tokens_string1=[" ".join(i) for i in tokens]
    tokens_string2=[" ".join(tokens_string1)]
    tokens3 = [word for sent in sent_tokenize(str(tokens_string2)) for word in word_tokenize(sent)]
    return tokens3

def frequent_200_words(tokens):
    # steps to generate the most frequent 200 words within the give data base
    # filtering out puctuations
    a=[w for w in tokens if w not in string.punctuation]
    # filtering out stop words
    filtered_words = [w for w in a if not w in stop_words]
    # calculate the frequence of each words and get the 200 most frequent words
    all_words = nltk.FreqDist(w.lower() for w in filtered_words)
    word_features = [i for (i,w) in all_words.most_common(200)]
    return word_features


def frequent_word_features(file, words): 
    # function to extract features of containing each of the 200 most frequent words or not
    file_words = set(file) 
    features = {}
    for word in words:
        features['contains({})'.format(word)] = (word in file_words)
    return features

def binary_features(featuresets):
    #generate feature set 
    #featuresets = [(frequent_word_features(n,words), label) for (n, label) in labeled_data]
    #vectorize feature set with DictVectorizer from sklearn
    vec1 = DictVectorizer(sparse=False)
    X = vec1.fit_transform([i for (i,w) in featuresets])
    # get Y vector
    Y = [w for (i,w) in featuresets]
    return X, Y

def run_adaboost_classifier(feature):
    X, Y=feature(data)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    #build adaboost classifier with count feature
    model = AdaBoostClassifier()
    model.fit(Xtrain, Ytrain)
    predictions = model.predict(Xtest)
    accuracy = accuracy_score(predictions, Ytest)
    print("classification rate for AdaBoost :", model.score(Xtest, Ytest),"accuracy:", accuracy)
    print(classification_report(Ytest,predictions))
    
def run_MultinomialNB_classifier(feature): 
    X, Y=feature(data)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    model = MultinomialNB()
    model.fit(Xtrain, Ytrain)
    print("classification rate for MultinomialNB :", model.score(Xtest, Ytest))
    predictions = model.predict(Xtest)
    print(classification_report(Ytest,predictions))
    
def run_MLP_Classifier(feature):
    X, Y=feature(data)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    model = MLPClassifier(hidden_layer_sizes=(30,30,30))
    model.fit(Xtrain, Ytrain)
    print("classification rate for Multi-Layer Perceptron Classifier :", model.score(Xtest, Ytest))
    predictions = model.predict(Xtest)
    print(classification_report(Ytest,predictions))
    

if __name__ == '__main__':

    data = load_data()
    print("with count feature, ") 
    run_adaboost_classifier(count_features)
    run_MultinomialNB_classifier(count_features)
    run_MLP_Classifier(count_features)
    print("with TFIDF feature, ") 
    run_adaboost_classifier(TFIDF_features)
    run_MultinomialNB_classifier(TFIDF_features)
    run_MLP_Classifier(TFIDF_features)
    print("with 5-gram charater cross word bound, ") 
    run_adaboost_classifier(ngram_char_crossW_features)
    run_MultinomialNB_classifier(ngram_char_crossW_features)
    run_MLP_Classifier(ngram_char_crossW_features)
    print("with 5-gram charater within word bound, ")
    run_adaboost_classifier(ngram_char_features)
    run_MultinomialNB_classifier(ngram_char_features)
    run_MLP_Classifier(ngram_char_features)
    print("with bigram word feature")
    run_adaboost_classifier(Bigram_word_features)
    run_MultinomialNB_classifier(Bigram_word_features)
    run_MLP_Classifier(Bigram_word_features)
    #use self difined binary feature
    
    #create labeled data{(data,label)}
    labeled_data=create_labeled_data(data)
    #tokenize all the words from data set
    tokens=tokenize_labeled_data(labeled_data)
    #extract the most frequent 200 words
    word_feature=frequent_200_words(tokens)
    #generate feature set of the most frequent 200 words
    featuresets = [(frequent_word_features(n,word_feature), label) for (n, label) in labeled_data]
    #generate X,Y matrix
    X,Y=binary_features(featuresets)
    #split into train and test set
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    #use self difined binary feature to run AdaBoost model
    model = AdaBoostClassifier()
    model.fit(Xtrain, Ytrain)
    print("with self difined binary feature, ", )
    print("classification rate for AdaBoost :", model.score(Xtest, Ytest))
    predictions = model.predict(Xtest)
    print(classification_report(Ytest,predictions))
    #use self difined binary feature to run MultinomialNB model
    model = MultinomialNB()
    model.fit(Xtrain, Ytrain)
    print("classification rate for multinomialNB :", model.score(Xtest, Ytest))
    predictions = model.predict(Xtest)
    print(classification_report(Ytest,predictions))
    #use self difined binary feature to run MLP classifier model
    model = MLPClassifier(hidden_layer_sizes=(30,30,30))
    model.fit(Xtrain, Ytrain)
    print("classification rate for Multi-Layer Perceptron Classifier :", model.score(Xtest, Ytest))
    predictions = model.predict(Xtest)
    print(classification_report(Ytest,predictions))