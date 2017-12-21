
Before going any further, please first start download of the data folder from the google drive link (1) here or the switch drive link (2)after that. 
For switch, the password is '' withought the single quotes : 
1. request the author
2. request the author


The project_folder  should have the following hierarchy : 
/project_final
	/data
		/model_final
		/other folders
		- other files
	-run.py
	-other files 



To install configure your environment please follow the following steps. 
	1. Install conda (Anaconda) and make sure you have conda environment variables set up in your bash. 
	Then run the following commands on your bash / command prompt: 
	1. conda create -n ml python=3.5
	2. source activate ml
	3. pip install -r requirements.txt
	4. conda install pandas bokeh jupyter nltk scikit-learn h5py --name ml
	
**** If you are using windows, you have to install curses manually with the following command (choose according to your architecture):
	python -m pip install ./data/curses-2.2-cp35-none-win32.whl             # 64 bit architecture
	python -m pip install ./data/curses-2.2-cp35-none-win_amd64.whl 		# 32 bit architecture

****In case you are unsure if you followed the steps right, please do it again after running the following command : conda remove -n ml --all ***** 



Before running run.py, ensure data is downloaded and it's inside the project_final folder. 
Then navigate to the project_file folder and run the following 2 commands: 
 1. source activate ml
 2. python run.py 


 Description, purpose and the usage of each file in project_final is elucidated below : 
 *** for TA's use ***
 1. run.py :  
 			Description - 	**This script won't work without ./data folder from Google drive**
					         This program either trains the final model or 
					         it uses a pretrained model to generate prediction - 'submission.csv'.
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
					                         generate submission, it will take around 7-8 hours)
					         run.py --full (this will preprocess the dataset and then it will train 
					                        the model and then generate submission, it will take around
					                        8-9 hours and needs a computer with 
					                        atleast 16 gb of RAM or process will be killed by OS) 
					        run.py -h  - (this will show you  all the options possible with the script)


*** for internal use ***
2. preprocessing_final.py :
							Description - 	This script contains the helper functions which are used by run.py in
											preprocessing the raw data. TA's may evaluate it to understand how we processed the data. 

 *** for TA's use ****
3. results_and_discussions.ipynb : 
							Description - If TA's would like to see the results of our best model in tensorboard interactively and 
										  SEE WHAT WORDS SO THE EMBEDDINGS CONTAIN,  HOW DOES THE PCA OF EMBEDDINGS LOOK , HOW OUR 
										  TRAINING CURVES LOOKED LIKE AND HOW OUR NETWORK GRAPH LOOKED LIKE  , they are STRONG ENCOURAGED
										  to see this file. And it also provides additional graphs, plots and description about each model we
										  tried. 

*** for TA's use ****
4. Archived_word : 
					Description -  This folder contains all the notebooks we used , almost all of them , which show the code, 
									and have a small introduction in top for few models. This is for TA's to evaluate the veracity of 
									the results we have presented. **DISCLAIMER** : Most of the codes won't run since they need more data
								 	and additional libraries for processing, and due to space constrains and conflicts(we used different environment for 
								 	keras because it uses different Tensorflow whose tensorboard isn't showing Projector, similarly xgboost needs another) 
								 	,we arent providing the instructions  and data to run  them 
									but if deemed necessary by you, we would be glad to provide the data and instructions for full working. 
*** for internal use ****
5. data : 
			Description - If this folder is missing , please kindly download it from the link mentioned at the top of this readme . Without it, almost nothing
						  in any notebooks , run.py can be run. Once you have the ENTIRE FOLDER, it will have plots and images used in results_and_discussion,
						  it also has the model_final which contains the fully trained model with the metadata used for projecting embeddings. Should you choose 
						  to train your own model, it will also be saved here under the folder of 'model_train_new' . You can navigate to ./data/model_train_new 
						  and then type the following command in your bash " tensorboard --logdir ." using the same runtime (ml) and tesnsorboard will stream the 
						  learning real time to you. 

**** for TA's use , just in case ****
6. requirements.txt :
			Description - This file contains the export of the pip install packages. Its used primarily to install the correct version of tensorflow and tflearn which 
						  we need for our program run.py to run. You shouldn't face any problems using the install commands above, but if you do, please check here 
						  what version of package is not available for your architecture and try installing the closest possible version of that package.

****END****
