# LING131FinalProject
The project could be divided into three parts.

Part one deals with choosing a better classifier and feature extractor, by testing and analyzing different classifiers with different features using nltk and scikit-learn.

Part two involves the implementation of almost every part of our course without much use of machine learning or natural language processing modules: from text tokenization and stemming, through feature selection, to model construction. We also analyze a few data sets and compare them to the results and source code in part one. From the comparison we find a few ways to improve.

In part three, we write a program that interacts with a user through command line.

## Part One: Classification using sklearn models
This part is in the file 'analysis_sklearn.py'. This file contains five functions to use sklearn methods to extract features of count, TFIDF, bigram words,ngram characters within word bound and ngram characters accross word bound. 

Then there is a self defined method to extract binary features of whether the most frequent 200 words are in the certain file.
After that, are several functions to run three different models from sklearn.

This file can be directly run with'python3 analysis_sklearn.py'.

The printed out result of this commond should looks like the follows:

With count feature

    classification rate for Adaboost:(the accuracy here)
    classification report(table) for this combination of Adaboost model and count feature
    classification rate for multinomialNB: (the accuracy here)
    classification report(table) for this combination of multinomialNB model and count feature
    classification rate for Multi-Layer Perceptron Classifier: (the accuracy here)
    classification report(table) for this combination of Multi-Layer Perceptron Classifier model and count feature

and then the printed out result on the screen will repeat this format of results with other five features.


## Part Two: Simplified Modules used in Text Classification

### 2.1


### 2.2 Data pre-processing
The main skeleten of this part is in the file ```data_processing.py```. User can call ```dataPrepro(raw_text, y_enc)``` method to get a tokenized data "data_tokenized" in the form like below:
```text
[[feature1_value, feature2_value, feature3_value..., label],
  ...
[feature1_value, feature2_value, feature3_value..., label]]
```
in which each feature value defines whether a n-gram(unigram or bigram) is in the sentence.

The parameter ```raw_text``` is a list of raw SMS text which can be extract from the second column of the dataset ```spam.csv```.
The parameter ```y_enc``` is a list of label corresponding to the raw text which can be exrtact from the first column of the dataset ```spam.csv```. 

```stemmer.py``` is a Stemmer class that import into ```data_processing.py``` for stemming the data. We built this stemmer under the idea of Porter Stemming.


### 2.3 Spam Classification Using Self-written Models

This part is in the file `analysis_model.py`. To run this code, one must make sure that `numpy`, `pandas`, and `sklearn` are installed. These tools are used to pre-process the text data and extract features before creating classifiers.

The code could be run in the Pycharm IDE as well as command line. Just type `python3 analysis_model.py` and press Enter, and you will get a result.

After running the code, you should see in the commandline or IDE a result like this:
```text
Feature: Raw Count
Self-built NeuralNetwork Accuracy with Raw Count:  61.93895870736086
Self-built NaiveBayes Accuracy with Raw Count:  87.07360861759426
Feature: TF-IDF
Self-built NeuralNetwork Accuracy with Raw Count:  85.278276481149
Self-built NaiveBayes Accuracy with Raw Count:  87.07360861759426
```
There might be several warnings in between lines, but generally they do not affect the result. So please ignore them.


## Part Three: Client Code
