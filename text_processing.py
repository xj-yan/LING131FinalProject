import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
import nltk
from Stemmer import Stemmer

def dataPrepro(raw_text, y_enc):
    '''
    preprocess data, and tokenize
    arg:
        raw_test: pandes array, each line contains a text
        y_enc: pandas array, each line is the label(1 for spam, 0 for ham)
    returns:
        data_tokenized: a processed and tokenized data(numpy array),
                        in the form like below:
                        [[feature1_value, feature2_value, feature3_value..., label],
                        ...
                        [feature1_value, feature2_value, feature3_value..., label]]
                        each feature value defines whether a n-gram(unigram or bigram) is in the sentence
        processed: the preprocessed text
    '''

    # replace e-mail address, url, money symbol, phone number and number
    # with emailaddr, httpaddr, moneysymb, phonenum, and number
    print('step1: replace emal,url,money symbol,phone number,number with their classes...')
    processed = raw_text.str.replace(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b',
                             ' emailaddr ')
    processed = processed.str.replace(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)',
                                      ' httpaddr ')
    processed = processed.str.replace(r'£|\$', ' moneysymb ')
    processed = processed.str.replace(
        r'(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        ' phonenum ')
    processed = processed.str.replace(r'(\s)?\d+(\.\d+)?(\s|\.|\,|\d|\?)', ' num ')
    print('done')

    # remove punctuations
    print('step2: remove punctuations, spaces...')
    processed = processed.str.replace(r'[^\w\d\s]', ' ')
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
    # we use our redefined simplified Stemmer to stem
    print('step4: stemming...')
    simple_porter = Stemmer()
    def stemming(sent):
        sent = sent.split(' ')
        word_list = [simple_porter.stem(word) for word in sent]
        return ' '.join(word_list)
    processed = processed.apply(stemming)
    print('done')


    # replace some odd words by mannual concluded rules
    print('step5: replaced with mannual rules...')
    mannual_word_map = {
        'aaooooright':'alright',
        'aww':'aw',
        'awww':'aw',
        'baaaaaaaabe':'babe',
        'baaaaabe':'babe',
        'boooo':'boo',
        'buzzzz':'buzz',
        'daaaaa':'da',
        'ffffffffff':'f',
        'fffff':'f',
        'ffffuuuuuuu':'fu',
        'geeee':'gee',
        'geeeee':'gee',
        'hmm':'hm',
        'hmmm':'hm',
        'hmmmm':'hm',
        'latelyxxx':'late',
        'lololo':'lol',
        'loooooool':'lol',
        'lool':'lol',
        'looovvve':'love',
        'miiiiiiissssssssss':'miss',
        'mmm':'mm',
        'mmmm':'mm',
        'mmmmm':'mm',
        'mmmmmm':'mm',
        'mmmmmmm':'mm',
        'nooooooo':'no',
        'noooooooo':'no',
        'oooh':'ooh',
        'oooooh':'ooh',
        'ooooooh':'ooh',
        'pleassssssseeeeee':'please',
        'sooo':'soo',
        'soooo':'soo',
        'sooooo':'soo',
        'ummmmmaah':'nmma',
        'xxxxx':'xxxx',
        'xxxxxx':'xxxx',
        'xxxxxxx':'xxxx',
        'xxxxxxxx':'xxxx',
        'xxxxxxxxx':'xxxx',
        'xxxxxxxxxxxxxx':'xxxx',
    }
    def mannualReplace(sent):
        sent = sent.split(' ')
        word_list = []
        for word in sent:
            if(word in mannual_word_map.keys()):
                word_list.append(mannual_word_map[word])
            else:
                word_list.append(word)
        return ' '.join(word_list)
    processed = processed.apply(mannualReplace)
    print('done')


    # replace rare word with <unk>
    print('step6: replace rare words with <unk>...')
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
    sorted_list = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
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
    print('step7: add noise....')
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
        if i % 500 == 0:
            print('proceeding data',i,'to',min(i+499,len(processed)))
        sent = processed[i].split(' ')
        if y_enc[i] == 1:
            for s in spam_list:
                edit_dist = editDistance(sent, s)
                if (edit_dist > 0) and (edit_dist < 3):
                    index = random.randint(0, len(s)-1)
                    if index < len(sent):
                        sent[index] = s[index]
                    else:
                        sent.append(s[index])
                    processed[i] = ' '.join(sent)
                    break
        else:
            for s in ham_list:
                edit_dist = editDistance(sent, s)
                if (edit_dist > 0) and (edit_dist < 3):
                    index = random.randint(0, len(s)-1)
                    if index < len(sent):
                        sent[index] = s[index]
                    else:
                        sent.append(s[index])
                    processed[i] = ' '.join(sent)
                    break
    print('done')


    # then we begin to tokenize
    print('tokenizing...')
    # construct the mapping from n-grams to feature indecies
    n_gram_map = {}
    for sent in processed:
        cnt = 0
        sent = sent.split(' ')
        for n in [1, 2]:
            for i in range(len(sent)-n):
                gram = ' '.join(sent[i:i+n])
                if gram not in n_gram_map.keys():
                    n_gram_map[gram] = cnt
                    cnt += 1

    # print(len(n_gram_map)) #there are totaly 31493 n-grams

    # begin tokenizing
    data_tokenized = []
    for i in range(len(processed)):
        feature_vec = [0] * 31494
        sent = processed[i].split(' ')
        for n in [1, 2]:
            for i in range(len(sent) - n):
                gram = ' '.join(sent[i:i + n])
                feature_vec[n_gram_map[gram]] = 1
        feature_vec[-1] = int(y_enc[i])
        data_tokenized.append(feature_vec)
    data_tokenized = np.array(data_tokenized)

    print('done, the data size is:', data_tokenized.shape[0], 'the feature size is ', data_tokenized.shape[1] - 1)
    return data_tokenized, processed
