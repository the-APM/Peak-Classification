import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense

def conf_table(y_pred,y_test):
	check=y_pred==y_test
	true_pos=0
	true_rej=0
	false_al=0
	miss=0
	for i in range(len(check)):
		if(check[i]):
			if(y_test[i]==1):
				true_pos+=1
			else:
				true_rej+=1
		else:
			if(y_test[i]==1):
				miss+=1
			else:
				false_al+=1
	print("True Positives=",true_pos)
	print("False Alarms=",false_al)
	print("Misses=",miss)
	print("True Rejections=",true_rej)

feature_names=list(pd.read_csv("training_NN.csv",nrows=1))
x_train = pd.read_csv("training_NN.csv",usecols=[i for i in feature_names if i!='label'])
x_test = pd.read_csv("testing_NN.csv",usecols=[i for i in feature_names if i!='label'])
y_train=pd.read_csv("training_NN.csv",usecols=['label'])
y_test=pd.read_csv("testing_NN.csv",usecols=['label'])
sm=SMOTE()
x_train,y_train=sm.fit_resample(x_train,y_train)
x_train,y_train=shuffle(x_train,y_train,random_state=0)
clf =Sequential()
clf.add(Dense(units=40,kernel_initializer='uniform',activation='relu',input_dim=27))
clf.add(Dense(units=25,kernel_initializer='uniform',activation='relu'))
clf.add(Dense(units=15,kernel_initializer='uniform',activation='relu'))
clf.add(Dense(units=5,kernel_initializer='uniform',activation='relu'))
clf.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
clf.compile(optimizer='adadelta',loss='binary_crossentropy',metrics=['accuracy'])
clf.fit(np.stack(x_train.tolist()),y_train.tolist(),epochs=10)
##clf.fit(np.stack(x_train.values.tolist(),axis=0),y_train['label'].tolist(),epochs=10)
y_pred1 = clf.predict(np.stack(x_test.values.tolist(),axis=0))
print(y_pred1)
y_pred1=(y_pred1>0.5)
y_pred1=y_pred1.reshape(len(y_pred1))
y_test = y_test['label'].tolist()
print(clf.get_weights())
check1 = y_pred1==y_test
print ("percentage of samples classified correctly - ",(float(sum(check1))/float(len(check1)))*100)
conf_table(y_pred1,y_test)
##f_imp=clf.feature_importances_
##for i in range(len(f_imp)):
##	print(feature_names[i],"=",f_imp[i])
# train_df = pd.read_csv("training_NN.csv")
# test_df = pd.read_csv("testing_NN.csv")
