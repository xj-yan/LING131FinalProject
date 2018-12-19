import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from analysis_sklearn import ngram_char_features
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# stop_words = set(stopwords.words('english'))

def ngram_char_features_text(text):
    #calculate the ngram of characters within a word bound, 
    #parameter n give the number of ngram. calculate exact that number of gram, 
    #means not including ngrams less than n.
    ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(5, 5))
    text_transformed=ngram_vectorizer.fit_transform(text)                            
    return text_transformed

def run_MLP_Classifier(feature, text):
    X, Y=feature(data)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    model = MLPClassifier(hidden_layer_sizes=(30,30,30))
    model.fit(Xtrain, Ytrain)
    predictions = model.predict(text)
    print(predictions)

if __name__ == '__main__':
	data = pd.read_csv('spam.csv',encoding = "ISO-8859-1")

	print("This is a spam detector")
	print("You can input received msg to test whether this is a spam")
	text = input("Please enter you message: ")
	text = [text]
	# translator = str.maketrans('', '', string.punctuation)
	# test = test.translate(translator)
	# print(test)
	# word_token = test.split(' ')
	# stop_words = set(stopwords.words('english'))
	text = ngram_char_features_text(text)
	run_MLP_Classifier(ngram_char_features, text)