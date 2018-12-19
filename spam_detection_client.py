import string
from nltk.corpus import stopwords

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