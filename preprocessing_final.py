
# coding: utf-8

# # Preprocessing

#Imports
import pickle
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('wordnet')


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
    token=token.replace("isn't", 'is not')
    token=token.replace("don't",'do not')
    token=token.replace("tis", ' it is')
    token=token.replace("'d", " would")
    token=token.replace("'ll"," will")
    token=token.replace("'s"," is")
    token=token.replace("twas", ' it was')
    token=token.replace("shan't", ' shall not')
    #rest: replace n't by not (with leading space)
    token=token.replace("n't", " not")
    
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


class Preprocess():
    
    

    
    def __init__(self):
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.words_expand_dict = pickle.load(open('./data/expand_join_dict_full','rb'))
        self.slang_dict = pickle.load(open('./data/'+'slang_dict.pkl','rb'))
    

    def preprocess_tweet(self,tweet, handles=True, length=True, negative=True, stop=False, 
                         numbers=True, urls=True, slang=True, expand_hash=True,path_slang='./data/'):
        '''
        Preprocess tweet with different possibilities. 
        INPUT:
            tweet: string with the tweet 
            handles: if true, remove handles
            length: if true, reduce length of more than three characters repeated to three characters, e.g, cooool -> coool
            negative: if true, replace contractions by not, e.g. can't -> can not
            stop: if true, remove stopwords
            numbers: if true, if true, remove numbers
            urls: if true, remove urls
            slang: if true, replace slang words
            path_slang: path to the pickled slang dictionary
        OUTPUT:
            string with preprocessed tweet
        '''
        #Convert tweet to tokens, remove handles and reduce length
        tknzr = TweetTokenizer(strip_handles=handles, reduce_len=length)
        tweet=tknzr.tokenize(tweet)
        
	
        #Replace negative contractions by not
        if negative:
        	    output=[]
        	    for t in tweet:
            		output.append(t)
            		to_replace=replace_negative(t)
            		if(to_replace != t):
            			replace_tweets=tknzr.tokenize(t)
            			output.extend(replace_tweets)
        	    tweet=output
            
        tweet = [ self.wordnet_lemmatizer.lemmatize(token) for token in tweet ]
        
        #Remove numbers
        if numbers:
            tweet = [token for token in tweet if not (token.isdigit() 
                                             or token[0] == '-' and token[1:].isdigit())]
            
        #Replace slang
        if slang:
            #Load slang dictionary
            #replace slang
            tweet=[replace_slang(token, self.slang_dict) for token in tweet]
         
        if(expand_hash):
            
            mult_char={}
            for char_ in 'abcdefghijklmnopqrstuvwxyz':
                mult_char[char_]=char_*3
            output=[] 
        
            for token in tweet:
                feature=token
                if('#' in feature):
                    feature=feature.replace('#','')                        
                    feature=feature.replace('"',"")
                    feature=feature.replace("'","")
                    feature=feature.replace("-","")
                    feature=feature.replace("_","")
                    for key,value in mult_char.items():
                        feature = feature.replace(value,key)
                        output.append(token)
                output.append(feature)
                if (feature in self.words_expand_dict):
                    output.extend(self.words_expand_dict[feature])
                
            tweet=output
            
        
        #Return a string and not tokens
        return " ".join(tweet)
    
def read_tweet_to_list(positive,full,is_train=True,max_num=None):
    '''
    Read tweets from the database
    INPUT:
        positive: boolean, if true, read the positive tweets file
        full: boolean, if true, read the tweets file with all tweets
        is_train: boolean, if true, read labeled tweets
        max_num: integer, maximum number of tweets to read
    OUTPUT:
        list with tweets read
    '''
    tweets=[]
    base='train_'
    if positive:
        base+='pos'
    else:
        base+='neg'
    if full:
        base+='_full'
    if(is_train==False):
        base='test_data'
    base+='.txt'
    count=0
    max_len=0
    with open('./data/{}'.format(base)) as f:
        for line in f:
            if(len(line)>max_len):
                max_len=len(line)
            tweets.append(''.join(list(filter(lambda x: not x.isdigit(), line))))
            if(max_num):
                count+=1
                if(count>max_num):
                    break
    print ('max len of is_positive:{} tweet is {}'.format(positive,max_len))
    return tweets
    
def generateX_Y(data, preprocessing_options={}):
    '''
    Preprocess and generate features for the data, and return X and Y 
    
    INPUT:
        data: dataframe with data. Y values are in column smile, and tweets in column Tweet
        preprocessing_options: dictionary with arguments for preprocessing_tweet function
    OUTPUT:
        features of X, Y, number of features, features, and vectorizer
    '''
    X, Y = data.drop('smile',1),data['smile']
    pp=Preprocess()
    X['tweet'] =X['tweet'].apply(pp.preprocess_tweet, **preprocessing_options)
    Y = to_categorical(Y,nb_classes=2)
    tknzr = TweetTokenizer()
    vectorizer = CountVectorizer(tokenizer=tknzr.tokenize,min_df=5)
    bag_trainX = vectorizer.fit_transform(X['tweet'])
    features=vectorizer.get_feature_names()
    num_vocab = len(vectorizer.get_feature_names())

    bag_of_ids_trainX = {"bag": [], 'max_len': bag_trainX.shape[1]}
    for bag in bag_trainX:
        bag_of_ids_trainX['bag'].append(np.where(bag.toarray() > 0)[1])
    bag_of_ids_trainX['bag'] = pad_sequences(bag_of_ids_trainX['bag'], maxlen=100)

    return bag_of_ids_trainX['bag'], Y,num_vocab,features,vectorizer

def divide_train_test(df_full, prep_option, num_train_tweets):
    '''
    Divide data in train, validation and test  sets
    
    INPUT:
        df_full: dataframe with all data, first training and then test set
        num_train_tweeets: number of tweets in training set
        prep_option: dictionary with arguments for preprocessing_tweet function
    OUTPUT:
        trainX, trainY, valX, valY, testX, number of features, features
    
    '''
    X, Y,num_vocab,features,vectorizer = generateX_Y(df_full, prep_option)
    X_test=X[num_train_tweets:]
    X=X[:num_train_tweets]
    Y=Y[:num_train_tweets]
    #Permute
    permute = np.random.permutation(len(X))
    X = X[permute]
    Y = Y[permute]
    fraction_test=0.8
    trainX , trainY = X[:int(fraction_test*(len(X)))],Y[:int(fraction_test*(len(Y)))]
    testX , testY = X[int(fraction_test*(len(X))):],Y[int(fraction_test*(len(Y))):]
    
    return trainX,trainY,testX,testY,X_test,num_vocab, features


