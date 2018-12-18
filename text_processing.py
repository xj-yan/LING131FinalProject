import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
import nltk


# we store the label of data in y_enc, and raw text in raw_text
# the final processed is the raw_text after wbeing processed
random.seed(2018)
df = pd.read_csv('spam.csv', header=0, sep = ',', encoding='ISO-8859-1')
y = df['v1']
le = LabelEncoder()
y_enc = le.fit_transform(y)
raw_text = df['v2']


# replace e-mail address, url, money symbol, phone number and number with emailaddr, httpaddr, moneysymb, phonenum, and number
print('step1: replace emal,url,money symbol,phone number,number with their classes...')
processed = raw_text.str.replace(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b',
                         ' emailaddr ')
processed = processed.str.replace(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)',
                                  ' httpaddr ')
processed = processed.str.replace(r'£|\$', ' moneysymb ')
processed = processed.str.replace(
    r'(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
    ' phonenum ')
processed = processed.str.replace(r'(\s)?\d+(\.\d+)?(\s|\.|\,|\d|\?)', ' number ')
print('done')


# remove punctuations and spaces
print('step2: remove punctuations, spaces...')
processed = processed.str.replace(r'[^\w\d\s]', ' ')
processed = processed.str.replace(r'\s+', ' ')
processed = processed.str.replace(r'^\s+|\s+?$', '')
processed = processed.str.lower()
print('done')


# remove stop words
# here we define an inline function removeStopWord to generate text without stopwords
print('step3: remove stop words...')
stop_words = set(nltk.corpus.stopwords.words('english'))
def removeStopWord(sent):
    sent = sent.split(' ')
    word_list = [word for word in sent if word not in stop_words]
    return ' '.join(word_list)
processed = processed.apply(removeStopWord)
print('done')


# stemming
# here we define an inline function stemming to chops off the ends of the words and make sure words with same meaning look the same
print('step4: stemming...')
porter = nltk.PorterStemmer()
def stemming(sent):
    sent = sent.split(' ')
    word_list = [porter.stem(word) for word in sent]
    return ' '.join(word_list)
processed = processed.apply(stemming)
print('done')


# replace rare word with <unk>
print('step5: replace rare words with <unk>...')
# replace number again
processed = processed.str.replace(r'\s\d+(\.\d+)?(\s|\.|\,|\d|\?)', ' number ')

vocab = {}
# building inventory
for sent in processed:
    words = sent.split(' ')
    for word in words:
        if(word not in vocab.keys()):
            vocab[word] = 1
        else:
            vocab[word] += 1
# sorted words by their frequency, from high to low
sorted_list = sorted(vocab.items(), key = lambda x: x[1], reverse = True)
# print(sorted_list[:-1000])
preserved_list = []
for i in range(len(sorted_list)):
    preserved_list.append(sorted_list[i][0])
# print('size of vocab:',len(preserved_list))
# preserve the first 6000 words in preserved_list
preserved_list = preserved_list[:6000]

def replaceUNK(sent):
    sent = sent.split(' ')
    for i in range(len(sent)):
        if(sent[i] not in preserved_list):
            sent[i] = '<unk>'
    return ' '.join(sent)
processed = processed.apply(replaceUNK)
print('done')


# To avoid over fitting, add some noise to the modal to increase robustness
print('step6: add noise....')
spam_list = []
ham_list = []
# seperate our current data to ham and spam list
for i in range(len(processed)):
    if(y_enc[i] == 1):
        spam_list.append(processed[i].split(' '))
    else:
        ham_list.append(processed[i].split(' '))

# using dynamic programming to define a function to calculate edit distance
def editDistance(l1,l2):
    len1 = len(l1) + 1
    len2 = len(l2) + 1
    # create matrix
    e = [0 for n in range(len1 * len2)]
    # first row of the matrix
    for i in range(len1):
        e[i] = i
    # first coloum of the matrix
    for j in range(0, len(e), len1):
        if(j % len1 == 0):
            e[j] = j // len1
    # get edit distance by state transit formula
    for i in range(1,len1):
        for j in range(1,len2):
            if l1[i-1] == l2[j-1]:
                cost = 0
            else:
                cost = 1
            e[j*len1+i] = min(e[(j-1)*len1+i]+1,
                    e[j*len1+(i-1)]+1,
                    e[(j-1)*len1+(i-1)] + cost)
    return e[-1]

# processing data
for i in range(len(processed)):
    if(i % 500 == 0):
        print('proceeding data',i,'to',min(i+499,len(processed)))
    sent = processed[i].split(' ')
    if(y_enc[i] == 1):
        for s in spam_list:
            edit_dist = editDistance(sent,s)
            if((edit_dist > 0) and (edit_dist < 3)):
                index = random.randint(0,len(s)-1)
                if(index < len(sent)):
                    sent[index] = s[index]
                else:
                    sent.append(s[index])
                processed[i] = ' '.join(sent)
                break
    else:
        for s in ham_list:
            edit_dist = editDistance(sent,s)
            if((edit_dist > 0) and (edit_dist < 3)):
                index = random.randint(0,len(s)-1)
                if(index < len(sent)):
                    sent[index] = s[index]
                else:
                    sent.append(s[index])
                processed[i] = ' '.join(sent)
                break
print('done')
