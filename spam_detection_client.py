import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from analysis_sklearn import ngram_char_features, create_labeled_data, tokenize_labeled_data

stop_words = set(stopwords.words('english'))



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
	test = input("Please enter you message: ")
	translator = str.maketrans('', '', string.punctuation)
	test = test.translate(translator)
	print(test)
	word_token = test.split(' ')
	stop_words = set(stopwords.words('english'))
	print(type(word_token))
	print(word_token)