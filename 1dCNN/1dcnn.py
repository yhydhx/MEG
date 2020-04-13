from keras.datasets import mnist
from keras.utils import to_categorical

from keras import layers
from keras import models
import numpy as np
import random
from keras.layers.convolutional import Conv1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import MaxPooling1D

def load_data(subj_id,duration):
	trainset,traintarget,testset,testtarget = [],[],[],[]
	
	raw_data = np.load("./subj"+subj_id+"/"+duration+"/y_norm.npy")
	data = []
	for line in raw_data:
		data.append(line[0])

	print(np.shape(raw_data))
	result_list = []
	for line in open("./subj"+subj_id+"/condid_label.csv"):
		filename = line.strip()
		result_list.append(filename)

	count = 0
	for single_data in data:

		#print single_label,single_data
		if random.random()< 0.75:
			trainset.append(single_data)
			traintarget.append(result_list[count])
		else:
			testset.append(single_data)
			testtarget.append(result_list[count])
		count += 1
	
	trainset = np.array(trainset)
	traintarget = np.array(traintarget)
	testset = np.array(testset)
	testtarget = np.array(testtarget)
	return trainset,traintarget,testset,testtarget




def cnn(subj_id, duration):
	(train_images, train_labels, test_images, test_labels) =  load_data(subj_id,duration)

	print(train_images.shape)
	n_timesteps, n_features, n_outputs = train_images.shape[0], train_images.shape[1], train_labels.shape[0]
	train_images = train_images.reshape(n_timesteps,n_features,1)
	train_images = train_images.astype('float32') / 255
	test_images = test_images.reshape(test_images.shape[0], 32,1)
	test_images = test_images.astype('float32') / 255

	train_labels = to_categorical(train_labels)
	test_labels = to_categorical(test_labels)

	model = models.Sequential()
	model.add(Conv1D(filters=32, kernel_size=3, input_shape=(32, 1)))
	model.add(Conv1D(filters=64, kernel_size=3))
	model.add(Conv1D(filters=128, kernel_size=3))
	model.add(Conv1D(filters=256, kernel_size=3))
	#model.add(Conv1D(filters=512, kernel_size=5))
	model.add(MaxPooling1D(pool_size=5 ))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(6, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.fit(train_images, train_labels,
	          batch_size=32,
	          epochs=30,
	          verbose=1)

	test_loss, test_acc = model.evaluate(test_images, test_labels)
	print('Test accuracy:', test_acc)
	return test_acc


def main():
	
	acc = {}
	for i in [3]:
	#for i in [1,2,3,4,5,6,7,8,9,11,12,13,15,18,19,20]:

		index = str(i)
		duration1 = "60-120"
		acc1 = cnn(index,duration1)
		duration2 = "60-200"
		acc2 = cnn(index,duration2)
		duration3 = "120-200"
		acc3 = cnn(index,duration3)

		acc[index] = {duration1:acc1,duration2: acc2, duration3:acc3}
	print(acc)
main()

