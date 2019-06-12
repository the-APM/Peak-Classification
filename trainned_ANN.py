import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("filepath",help="Takes the testing file path")
arg=parser.parse_args()
testing_file=arg.filepath
feature_names=list(pd.read_csv(testing_file,nrows=1))
x_test = pd.read_csv(testing_file,usecols=[i for i in feature_names if i not in ['GroupId','label']])
y_test=pd.read_csv(testing_file,usecols=['label'])
gid_test=pd.read_csv(testing_file,usecols=['GroupId'])
x_scale=preprocessing.RobustScaler().fit(x_test)
x_test=x_scale.transform(x_test)

with tf.Session() as session:
	clf =Sequential()
	clf.add(Dense(units=30,kernel_initializer='uniform',activation='relu',input_dim=20))
	clf.add(Dense(units=20,kernel_initializer='uniform',activation='elu'))
	clf.add(Dense(units=10,kernel_initializer='uniform',activation='elu'))
	clf.add(Dense(units=2,kernel_initializer='uniform',activation='softmax'))
	clf.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
	saver=tf.train.Saver()
	saver.restore(session,"./Trained_ANN/trained_ANN")
	y_pred1 = clf.predict(np.stack(x_test.tolist()))

y_pred=[]
for i in range(len(y_pred1)):
	y_pred.append(np.where(y_pred1[i]==max(y_pred1[i]))[0][0])
print(y_pred)
