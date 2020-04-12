#data collection
#RAE
import keras
from keras import backend as K
from keras.utils import Sequence
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives
from keras import regularizers
from sklearn.linear_model import Lasso

import numpy as np
import scipy.io
np.set_printoptions(precision = 2, suppress = True)
import os
#import nibabel as nib
import pandas as pd
#from nibabel import cifti2 as ci
from scipy import stats

import matplotlib.pyplot as plt
from numpy import *





def RAE(subj_id,duration,LENGTH):
	DATA_NUM = 1
	
	DATA_LENGTH = LENGTH*306 #10000 #2587647
	data = np.load("subj"+subj_id+"/data_cut_"+duration+".npy")
	#event_list = np.load("events.npy")
	print(np.shape(data))
	TIME_LENGTH = np.shape(data)[0] 
	

	sub = np.memmap('sub_Gambling.mymemmap', dtype='float32', mode='r+', shape=(TIME_LENGTH * DATA_NUM, DATA_LENGTH))
	sub_data = np.memmap('sub_Gambling.mymemmap', dtype='float32', mode='r+', shape=(TIME_LENGTH * DATA_NUM, DATA_LENGTH))

	sub[0 * TIME_LENGTH:1 * TIME_LENGTH, ] = data[0 * TIME_LENGTH:1 * TIME_LENGTH, ]

	### zscore y#####

	sub = np.memmap('sub_Gambling.mymemmap', dtype='float32', mode='r+', shape=(TIME_LENGTH * DATA_NUM,DATA_LENGTH))

	sub_data = np.memmap('sub_Gambling.mymemmap', dtype='float32', mode='r+', shape=(TIME_LENGTH * DATA_NUM,DATA_LENGTH))
	
	print(np.shape(sub))
	img_step = TIME_LENGTH-1 #volumeLength-1
	cnt = 0
	for num2 in range(0,DATA_NUM*(img_step+1)):
	    cnt += 1
	    if cnt == (img_step + 1):
	        cnt = 0
	        sub_data[num2 - img_step: num2 + 1,:] = stats.zscore(sub[num2 - img_step: num2 + 1,:])

		############# many to many ##########
		#RAE
	
	sub_data = np.memmap('sub_Gambling.mymemmap', dtype='float32', mode='r+', shape=(TIME_LENGTH*DATA_NUM,DATA_LENGTH))
	# print(sub_data)
	img_step = TIME_LENGTH - 1
	sub_data1 = sub_data[0:DATA_NUM*(img_step+1),:]
	print(np.shape(sub_data1))
	data = np.expand_dims(sub_data1, axis=1)
	data = np.reshape(data, (DATA_NUM,TIME_LENGTH , DATA_LENGTH))
	# latent_dim = [320]


	input_dim = data.shape[-1] # 13
	timesteps =  data.shape[1]# 3

	inputs = Input(shape=(timesteps, input_dim,))
	layer1 = Dense(128, activation='tanh',activity_regularizer=regularizers.l1(1*10e-7), kernel_regularizer=regularizers.l2(1*10e-4))#, activity_regularizer=regularizers.l1(1*10e-7), kernel_regularizer=regularizers.l2(1*10e-4)
	encoded = layer1(inputs)
	encoded = LSTM(64,return_sequences=True,activity_regularizer=regularizers.l1(1*10e-7), kernel_regularizer=regularizers.l2(1*10e-4))(encoded)#
	encoded = LSTM(32,return_sequences=True,activity_regularizer=regularizers.l1(1*10e-7), kernel_regularizer=regularizers.l2(1*10e-4))(encoded)#, activity_regularizer=regularizers.l1(1*10e-7), kernel_regularizer=regularizers.l2(1*10e-4)

	encoder = Model(inputs, encoded)
	encoder.summary()

	inputs = Input(shape=(timesteps, 32,))
	decoded = LSTM(64, return_sequences=True,activation='tanh')(inputs)
	decoded = LSTM(128, return_sequences=True,activation='tanh')(decoded)
	outputs = Dense(input_dim, activation='tanh')(decoded)


	decoder = Model(inputs, outputs)
	decoder.summary()
	# model for RAE
	inputs = Input(shape=(timesteps, input_dim,))
	outputs = encoder(inputs)
	outputs = decoder(outputs)
	sequence_autoencoder = Model(inputs, outputs)

	# autoencoder = Model(inputs, decoded)

	sequence_autoencoder.compile(optimizer='adam', loss='mse')
	sequence_autoencoder.summary()
	sequence_autoencoder.fit(data, data, epochs=25, batch_size =1)

	#### predict hidden layer ###

	y = np.zeros((DATA_NUM,TIME_LENGTH,32), dtype=float)
	for i in range(0,DATA_NUM):
	    y[i * 1 : (i + 1) * 1]=encoder.predict(data[i * 1 : (i + 1) * 1])

		#print(y)

	### zscore y#####
	
	print(y.shape)
	y = np.reshape(y, (DATA_NUM*TIME_LENGTH, 1, 32))
	t = range(0, TIME_LENGTH)
	plt.plot(t, y[TIME_LENGTH*0: TIME_LENGTH*1, 0,1])
	y_norm = np.zeros((TIME_LENGTH*DATA_NUM, 1,32), dtype=float)
	img_step = TIME_LENGTH-1
	cnt = 0
	for num2 in range(np.shape(y)[0]):
	    cnt += 1
	    if cnt == (img_step + 1):
	        cnt = 0
	        y_norm[num2 - img_step: num2 + 1,:] = stats.zscore(y[num2 - img_step : num2 + 1,:])
	        

	where_are_NaNs = isnan(y_norm)
	y_norm[where_are_NaNs] = 0
	plt.plot(t, y_norm[TIME_LENGTH*0: TIME_LENGTH*1, 0,1])

	folder_name = "./subj"+subj_id+"/"+duration+"/"
	print("========================")
	print(data[0][0])
	#np.save(folder_name+"data_before.npy",data)
	print("========================")
	print(y_norm[0])
	np.save(folder_name+"y_norm.npy",y_norm)
	print("========================")
	print(y[0])
	np.save(folder_name+"y.npy",y)
	print("========================")
	#data_predicted = sequence_autoencoder.predict(data)
	#np.save(folder_name+"data_predicted.npy",data_predicted)


def main():
	for i in [2,3,4,5,6,7,8,9,11,12,13,15,18,19,20]:
		print(i)
		RAE(str(i),"60-120",60)
		RAE(str(i),"60-200",140)
		RAE(str(i),"120-200",80)

main()