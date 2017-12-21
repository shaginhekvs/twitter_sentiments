#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 02:52:37 2017

@author: keshav
"""

import argparse,textwrap,os
import pandas as pd
import tflearn as tf
import numpy as np
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tensorflow.contrib.tensorboard.plugins import projector
import pickle
from preprocessing_final import read_tweet_to_list , divide_train_test


def preprocess_raw_files(full_file=False):
    '''
    Preprocessing of the files and save the files in data/processed_data
    INPUT:
        full_file: if true, read the whole dataset of 2500000 tweets
    
        
    '''
    hours = 0.25
    print('preproprocessing input text files. It can take 1-2 hours')
    print('reading neg tweets')
    neg_tweets_2=read_tweet_to_list(False,False)
    print('reading pos tweets')
    pos_tweets_2=read_tweet_to_list(True,False)
    print('reading test tweets')
    test_tweets=read_tweet_to_list(False,False,False,max_num=None)
    
    df_neg_tweets=pd.DataFrame({'tweet':neg_tweets_2,'smile':list(np.zeros(len(neg_tweets_2)))})
    df_pos_tweets=pd.DataFrame({'tweet':pos_tweets_2,'smile':list(np.ones(len(pos_tweets_2)))})
    
    df_test_tweets=pd.DataFrame({'tweet':test_tweets,'smile':list(np.ones(len(test_tweets)))})
    
    
    if(full_file):
        hours = 1
        neg_tweets=read_tweet_to_list(False,True,max_num=None)
        pos_tweets=read_tweet_to_list(True,True,max_num=None)
        df_neg_tweets=pd.DataFrame({'tweet':neg_tweets+neg_tweets_2,'smile':list(np.zeros(len(neg_tweets)+len(neg_tweets_2)))})
        df_pos_tweets=pd.DataFrame({'tweet':pos_tweets+pos_tweets_2,'smile':list(np.ones(len(pos_tweets)+len(pos_tweets_2)))})

    print('reading completed, preprocessing now, wait for additional {} hour without any message'.format(hours))
    
    df_full=pd.concat([df_pos_tweets,df_neg_tweets,df_test_tweets])    
    prep_opts=dict(handles=True, length=True, negative=True, stop=False, 
                         numbers=True, urls=False, slang=True)
    
    num_train_tweets=len(df_neg_tweets)+len(df_pos_tweets)
    trainX,trainY,testX,testY,X_test,num_vocab, features=divide_train_test(df_full, prep_opts, num_train_tweets)       
    pickle.dump([trainX,trainY,testX,testY,X_test,num_vocab,features],open('./data/processed_data','wb'))

    

def create_model(num_vocab,train=False,embedding_matrix=None,model_path='./data/model_final/model_saved.ckpt'):
    '''
        	Creates model with two parallel branches: one with LSTM and another with CNN.
        INPUT:
            num_vocab: number of words in features
            train: if true, train the model from scratch. If false, load the pretrained model. DEFAULT: False
            embedding_matrix: embedding matrix to use with the Neural Net. DEFAULT: None
    		model_path: If train is false, path to the final model. DEFAULT: './data/model_final/model_saved.ckpt'
        OUTPUT:
            model
    	
    '''    
    print('initializing the model, will take around 5 mins')
    	
    	#Initialize the model 
    tensorflow.reset_default_graph()
    	
    	#Input data with padded samples until 100
    net = tf.input_data([None, 100] ,name='input_layer')
    	
    	#If not training, initalize with whatever
    if(embedding_matrix==None):   
        embedding_matrix = np.random.random((num_vocab,300))
    		
    	####### CNN #########
    	#Initalize weights
    W1 = tensorflow.constant_initializer(embedding_matrix)
    
    	#Embedding layer of size 300
    net1 = tf.embedding(net,input_dim=num_vocab,output_dim=300,weights_init= W1 , name='embedded_layer')
    #Convolutional layer with window size = 3
    net1 = tf.conv_1d ( net1, 2 , 3 , activation='relu',name='conv_layer')
    	#Select the most representative of the 3 neighbots
    net1 = tf.max_pool_1d ( net1 , 3 , strides = 1,name='max_pool_layer')
    
    
    #Add 0.8 (keep) dropout for overfitting
    net1 = tf.dropout(net1,0.8,name='conv_dropout_layer')
    net1 = tf.flatten(net1,name='flatter_conv_layer')
    	
    #Fully connected layer with ReLu
    net1 = tf.fully_connected(net1, 150, activation='relu',name='first_fc')
    
    ####### LSTM #########
    	#Initialize weigths
    W2 = tensorflow.constant_initializer(embedding_matrix)
    	#Embedding layer of size 300
    net2 = tf.embedding(net, input_dim=num_vocab,output_dim=300,weights_init= W1 ,name='embedded_layer2')
    	#LSTM with 0.8 dropout and 256 hidden cells
    net2 = tf.lstm(net2, 256, dropout = 0.8,name = 'LSTM_layer',return_seq=True)
    	
    	#If a list is returned (error with return_seq), stack them into a 3D matrix
    if isinstance(net2, list):
        net2 = tensorflow.stack(net2, axis=1,name='stack')
    
    	#Fully connected layer with ReLu
    net2 = tf.fully_connected(net2, 200, activation='relu',name='second_fc')
    
    	
    	###### MERGE LAYER ######
    	#Concatenate results from both branches
    net_final=tensorflow.concat([net1,net2],1, name="concat")
    
    	#Fully connected layer with ReLu
    net_final = tf.fully_connected(net_final, 128, activation='relu')
    
    	#Dropout of 0.8
    net_final = tf.dropout(net_final, 0.8,name='merge')
    	
    	###### OUTPUT  LAYER ######
    	#Final layer with softmax 
    net_final = tf.fully_connected(net_final, 2, activation='softmax',name='output')
    	#Backpropagation with Adam and output with categorical_crossentropy: two columns each with the probability of smile or not
    net_final = tf.regression(net_final, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')
    
    model = tf.DNN(net_final, tensorboard_verbose=0,checkpoint_path=None,tensorboard_dir='./data/')
    
    #if not training new model then load old saved model. 
    if( not train):
        model.load(model_path)

    return model


def make_prediction(model,X_test,submission_path='./submission.csv'):
    '''
	Make predictions for Kaggle 
	INPUT
		model: model to make the predictions
		X_test: testing data to predict its outputs
		submission_path: path to save the submission
	
    '''
    print('making predictions, will take around 5 mins')
	#Predict output from the data
    pred=model.predict(X_test)
	#Save the index with higher probability as class predicted
    preds_array=np.argmax(pred,axis=1)
	#Change 0 column to -1 for submission
    preds_array[preds_array==0]=-1
	#Generate csv
    pd.DataFrame({'Id':np.array(list(range(1,len(preds_array)+1))),'Prediction':preds_array}).to_csv(submission_path,index=False)
    print('predictions done')
    
def load_processed_data(only_test=True,path='./data/train_test_prep_expand_full_2',embedding_matrix_path = './data/embedding_matrix_2'):
    '''
	Load pre-processed data from pickled file
	
	INPUT:
		only_test: If true, only return  test data preprocessed. Else, return everything. DEFAULT:True
		path: path to the pickled file. DEFAULT: './data/train_test_prep_expand_full_2'
	OUTPUT:
		if only_test=True -> num_vocab,X_test
		else -> trainX,trainY,testX,testY,X_test,num_vocab, features
    '''
    print('loading dataset, will take around 5 mins')
	#Load the data
    trainX,trainY,testX,testY,X_test,num_vocab, features = pickle.load(open(path,'rb'))
    
	#Return test data
    if(only_test):
        return [num_vocab,X_test]
	#Return all data
    else:
        trainX=np.concatenate([trainX,testX[:485000]])
        trainY=np.concatenate([trainY,testY[:485000]])
        testX=testX[485000:]
        testY=testY[485000:]
        embedding_matrix=None
        if not (embedding_matrix_path == None):
            embedding_matrix = pickle.load(open(embedding_matrix_path,'rb')) 
        return [trainX,trainY,testX,testY,X_test,num_vocab, features,embedding_matrix]
    

def save_embedding_metadata(features ,path='./data/model_train_now/'):
    '''
    Save the metadata to be used by projector later. 
    Input :
        features - list of words 
        path - path where the tensorflow is going to save the 
                embedding tensor
    '''
    summary_writer = tensorflow.summary.FileWriter(path)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = 'embedded_layer/W'
    path_file=os.path.join(path,'metadata.tsv')
    pd.Series(features).to_csv(path_file,sep='\n',index=False,header=False)
    embedding.metadata_path = 'metadata.tsv'
    # Saves a configuration file that TensorBoard will read during startup.
    projector.visualize_embeddings(summary_writer, config)
    

def start_training_model(model,trainX,trainY,testX,testY,run_id='model_train_now'):
    '''
        Trains the DNN model for 2 epochs
    '''
    print('training now, will take around 6 hours for completion')
    model.fit(trainX, trainY,validation_set=(testX,testY), \
              run_id=run_id,n_epoch=2, batch_size=128,\
              show_metric=True,snapshot_step=1000,snapshot_epoch=True)


def save_new_model(model,path='./data/model_train_now/model_saved.ckpt'):
    '''
        saves the model locally. 
    '''
    model.save(path)
    



def main(train=False,short=False,full=False):
    '''
	main function
    '''
    if (not train) and (not full) and (not short):
		#case when you want to use pre trained model to generate prediction. 
        num_vocab,X_test=load_processed_data()        
		#Load preprocessed model
        model=create_model(num_vocab)
		#Make the predictions
        make_prediction(model,X_test)
        
    if(train):
        #case when you want to train and not preprocess
        trainX,trainY,testX,testY,X_test,num_vocab, features,embedding_matrix = load_processed_data(only_test=False)
        save_embedding_metadata(features)
        model = create_model(num_vocab = num_vocab , train=True,embedding_matrix=embedding_matrix)
        start_training_model(model,trainX,trainY,testX,testY)
        make_prediction(model,X_test)
        save_new_model(model)
        
    if(short):
        #case when you you want to preprocess smaller input dataset and train model on it and generate prediction
        preprocess_raw_files(full_file=False)
        trainX,trainY,testX,testY,X_test,num_vocab, features,embedding_matrix = load_processed_data(only_test=False,path = './data/processed_data',\
                                                                      embedding_matrix_path = None)
        save_embedding_metadata(features)
        model = create_model(num_vocab = num_vocab , train=True,embedding_matrix=embedding_matrix)
        start_training_model(model,trainX,trainY,testX,testY)
        make_prediction(model,X_test)
        save_new_model(model)
        
    if(full):
        #case when you want to preprocess complete input dataset and train model on it and generate prediction
        preprocess_raw_files(full_file=True)
        trainX,trainY,testX,testY,X_test,num_vocab, features,embedding_matrix = load_processed_data(only_test=False,path = './data/processed_data',\
                                                                      embedding_matrix_path = None)
        save_embedding_metadata(features)
        model = create_model(num_vocab = num_vocab , train=True,embedding_matrix=embedding_matrix)
        start_training_model(model,trainX,trainY,testX,testY)
        make_prediction(model,X_test)
        save_new_model(model)    

    return 0
    # ...

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='tweet_sentiment_analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
        ** This script won't work without ./data folder from Google drive**
         This program either trains the model or 
         it uses a pretrained model to generate prediction.
         The network comprises of 2 parallel branches, 
         one of LSTM and another of CNN, 
         using glove embeddings which have been trained further in training.
         --------------------------------
         You have the following allowed use cases:

         run.py  (this will use pretrained model to generate submission
                    in ./submission.csv) 
         run.py --train (this will use the preprocessed dataset by us to train 
                     the model and then generate submission , it can take 
                     upto 6-7 hours so please be sure of what you are doing)
         
         run.py --short (this will preprocess the smaller training dataset files
                         and then it will train the model and then 
                         generate submission, it will take around 7-8 hours
                         In order to reduce the time for preprocessing and the data
                         we need to upload , we have skipped the initialisation with
                         glove embedding)
         run.py --full (this will preprocess the dataset and then it will train 
                        the model and then generate submission, it will take around
                        8-9 hours and needs a computer with 
                        atleast 16 gb of RAM or process will be killed by OS.
                        In order to reduce the time for preprocessing and the data
                        we need to upload , we have skipped the initialisation with
                        glove embedding)
         
              
        '''))
    parser.add_argument('-t','--train', action='store_true',help='Add this argument to train with preprocessed dataset')
    parser.add_argument('-s','--short', action='store_true',help='Add this argument to compute smaller version of preprocessed dataset and train')
    parser.add_argument('-f','--full', action='store_true',help='Add this argument to compute full preprocessed dataset and train')
    
    args = vars(parser.parse_args())
    print(args)
    main(**args)
