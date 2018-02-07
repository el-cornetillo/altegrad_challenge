import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk import word_tokenize
import re
import nltk
# nltk.download('punkt')
import chardet
import itertools
import logging
from nltk.stem.wordnet import WordNetLemmatizer
import unicodedata
tqdm.pandas('my bar !')
import os
import sys
import inflect
from constants import *

tkz = nltk.data.load('tokenizers/punkt/english.pickle')
p = inflect.engine()

def to_singular(word, p):
    if (nltk.pos_tag([word])[0][1] == 'PRP') or (word == 'his'): return word
    x = p.singular_noun(word)
    if x==False: return word
    return x

def sum_list(lst):
    try: return np.sum(lst)
    except: return lst[0]


def strip_accents_unicode(s):
    try:
        s = unicode(s, 'utf-8')
    except:  # unicode is a default on python 3
        pass
    s = unicodedata.normalize('NFD', s)
    s = s.encode('ascii', 'ignore')
    s = s.decode("utf-8")
    return str(s)

def question_to_wordlist(text, remove_sw = False, stem_words=True):
    text = re.sub(r"€", " euro ", text)
    text = re.sub(r"₹", " rupee ", text)
    text = strip_accents_unicode(text.lower())
    text = re.sub(r",000,000", "m", text)
    text = re.sub(r",000", "k", text)
    text = re.sub(r".000.000", "m", text)
    text = re.sub(r".000", "k", text)
    text = re.sub(r"000 ", "k ", text)
    text = re.sub(r"000000 ", "m ", text)
    text = re.sub(r"′", "'", text)
    text = re.sub(r"’", "'", text)
    text = re.sub(r'agaist', 'against', text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"won't", "will not ", text)
    text = re.sub(r" cannot ", " can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"it's ", "it is ", text)
    text = re.sub(r"he's ", "he is ", text)
    text = re.sub(r"she's ", "she is ", text)
    text = re.sub(r"\'re ", " are ", text)
    text = re.sub(r"\'d ", " would ", text)
    text = re.sub(r"\'ll ", " will ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", " 911 ", text)
    text = re.sub(r" 9/11 ", " 911 ", text)
    text = re.sub(r" 9-11 ", " 911 ", text)
    text = re.sub(r" e-mail ", " email ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " america ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r" uk ", " england ", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"qoura", "quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r" kms ", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" upvote ", " up vote ", text)
    text = re.sub(r" downvotes ", " down votes ", text)
    text = re.sub(r" downvote ", " down vote ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r" bf ", " best friend ", text)
    text = re.sub(r" iii ", " 3 ", text) 
    text = re.sub(r" ii ", " 2 ", text)
    text = re.sub(r"gta v ", "gta 5 ", text) 
    text = re.sub(r" the united states of america ", " america ", text)
    text = re.sub(r" the united states ", " america ", text)
    text = re.sub(r" j k ", " jk ", text)
    text = re.sub(r"%", " percent ", text)
    text = re.sub(r"rupees ", " rupee ", text)
    text = re.sub(r" rs", " rupee ", text)
    text = re.sub(r"\$", " dollar ", text)
    text = re.sub(r"emoticon", " emoji ", text)
    text = re.sub(r"emoticons", " emoji ", text)
    text = re.sub(r"smileys", " emoji ", text)
    text = re.sub(r"smiley", " emoji ", text)
    text = re.sub(r"emojis", " emoji ", text)
    text = re.sub(r"[^A-Za-z0-9]", " ", text)


    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    # Optionally, remove stop words
    if remove_sw:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = word_tokenize(text)
        verbs_stemmer = WordNetLemmatizer()
        #temmed_words = [verbs_stemmer.lemmatize(verbs_stemmer.lemmatize(word.lower(),'n'), 'v') for word in text]
        stemmed_words = [to_singular(verbs_stemmer.lemmatize(word.lower(), 'v'), p) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text.split(' '))

def question_to_sentences(text, tokenizer = tkz, remove_sw = False):
    text = tokenizer.tokenize(text.strip())
    sentences = []
    
    for t in text:
        if(len(t) > 0):
            sentences.append(question_to_wordlist(t, remove_sw))
    return sentences


def process(data):
	sentences = []

	print("Parsing sentences from training set...")
	#Converting question1 to sentences for word2vec model
	for i in tqdm(range(0, len(data['q1']))):
	    try:
	        #Check for empty strings ""
	        if(not pd.isnull(data['q1'][i])):
	            sentences += question_to_sentences(data['q1'][i])
	    except:
	        try:
	            encoding = chardet.detect(data['q1'][i])['encoding']
	            sentences += question_to_sentences(data['q1'][i].decode(encoding))
	        except:
	            print(encoding)

	#Converting question2 to sentences for word2vec model
	for i in tqdm(range(0,len(data['q2']))):
	    try:
	        if(not pd.isnull(data['q2'][i])):
	            sentences += question_to_sentences(data['q2'][i])
	    except:
	        try:
	            encoding = chardet.detect(data['q2'][i])['encoding']
	            sentences += question_to_sentences(data['q2'][i].decode(encoding))
	        except:
	            print(encoding)

	return sentences


def main(path_train, path_test):
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)

    df_train['q1p'] = df_train.q1.fillna("").progress_apply(lambda s : sum_list(question_to_sentences(s)))
    df_train['q2p'] = df_train.q2.fillna("").progress_apply(lambda s : sum_list(question_to_sentences(s)))
    df_train.to_csv('train_processed.csv', index=None)
    df_test['q1p'] = df_test.q1.fillna("").progress_apply(lambda s : sum_list(question_to_sentences(s)))
    df_test['q2p'] = df_test.q2.fillna("").progress_apply(lambda s : sum_list(question_to_sentences(s)))

    
    df_test.to_csv('test_processed.csv', index=None)


if __name__ == '__main__':
    args = [str(i) for i in sys.argv[1:]]
    main(args[0], args[1])





