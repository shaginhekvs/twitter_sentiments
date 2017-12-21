
# coding: utf-8

# # Preprocessing

#Imports
USERPATH='/Users/laurieprelot' # SPECIFY IN THE README 


import pickle
import nltk
# -----------------------Can comment out this part after first run
#nltk.download('all')
#nltk.download("wordnet") #nltk.download("wordnet", "whatever_the_absolute_path_to_myapp_is/nltk_data/")
#--------------------------
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import re

def replace_negative(token):
    ''' 
    Replace negative contracted verbs (isn't, won't) by their not contracted forms (is not, will not)
    
    INPUT:
        token: string with the word
    OUTPUT:
        replaced word
    '''
    #special cases: won't (different form)
    token=token.replace("won't", "will not")
    #special cases: can't (n is part of the word), cannot (split so it has the same form as all others)
    token=token.replace("can't", "can not")
    token=token.replace("cannot", "can not")
    token=token.replace("can'", "can not")

    #rest: replace n't by not (with leading space)
    token=token.replace("n't", " not")
    token=token.replace("'nt", " not")
    
    return token    
    
def replace_slang(token, slang):
    '''
    Replace slangs words with dictionary
    INPUT:
        token: word to replace
        slang: slang dictionary
    OUTPUT:
        replaced word
    '''
    #if token in slang dictionary, replace it
    try:
        return slang[token]
    #token not in slang dictionary
    except:
        return token


class My_antonym_replacer(object):
    '''Replace antonyms in the following way : ['Today','young','people', 'are','not','wise']-> ['Today', 'young', 'people', 'are', 'foolish']'''
    def replace(self, word, pos=None):
        antonyms = set()
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms.add(antonym.name())
                    if len(antonyms) == 1:
                        return antonyms.pop()
                    else:
                        return None

    def replace_negations(self, sent):
        i, l = 0, len(sent)
        words = []
        while i < l:
            word = sent[i]
            if word == 'not' and i+1 < l:
                ant = self.replace(sent[i+1])
                if ant:
                    words.append(ant)
                    i += 2
                    continue
                    words.append(word)
                    i += 1
                    return words


def preprocess_tweet(tweet, handles=False, length=True, negative=True, stop=True, numbers=False,
                     urls=False, antonyms=True, slang=True, path_slang='./twitter-datasets/'):
    '''
    Preprocess tweet with different possibilities. 
    INPUT:
        tweet: string with the tweet 
        handles: if true, remove handles
        length: if true, reduce length of more than three characters repeated to three characters, e.g, cooool -> coool
        negative: if true, replace contractions by not, e.g. can't -> can not
        stop: if true, remove stopwords
        numbers: if true, remove numbers
        urls: if true, remove urls
        antonyms: if true, replace antonyms
        slang: if true, replace slang words
        path_slang: path to the pickled slang dictionary
    OUTPUT:
        string with preprocessed tweet
    '''
    #Convert tweet to tokens, remove handles and reduce length
    tknzr = TweetTokenizer(strip_handles=handles, reduce_len=length)
    tweet = tknzr.tokenize(tweet)
    
    #Replace negative contractions by not
    if negative: 
        tweet=[replace_negative(token) for token in tweet]
        
    #Replace stopwords
    if stop:
        #List of stopwords
        stop = stopwords.words('english') 
        #Do not remove negations
        stop.remove('not')
        #Remove elemnents that are in list of stopwords
        tweet = [token for token in tweet if token not in stop]
    
    #Remove numbers
    if numbers:
        tweet = [token for token in tweet if not (token.isdigit() 
                                         or token[0] == '-' and token[1:].isdigit())]
    #Remove URLS
    if urls:
        #With regexp
        tweet=[re.sub(r'http\S+', '', token) for token in tweet]
        #If mentioned as <url>
        tweet=[token.replace("<url>", "") for token in tweet]
        
    #Replace slang
    if slang:
        #Load slang dictionary
        slang_dict=pickle.load(open(path_slang+'slang_dict.pkl','rb'))
        #replace slang
        tweet=[replace_slang(token, slang_dict) for token in tweet]
    
    #Replace antonyms 
    if antonyms:
        replacer = My_antonym_replacer()
        tweet=replacer.replace_negations(tweet)

    #Return a string and not tokens
    return " ".join(tweet)
    


