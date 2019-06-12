import pandas as pd
from sklearn import svm
import collections
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
x_train= pd.read_csv("training_NN.csv",usecols=[i for i in feature_names if i!='label'])
x_test= pd.read_csv("testing_NN.csv",usecols=[i for i in feature_names if i!='label'])
y_train=pd.read_csv("training_NN.csv",usecols=['label'])
y_test=pd.read_csv("testing_NN.csv",usecols=['label'])
counter=collections.Counter(y_train['label'].tolist())
impurity=counter[0]/(counter[1]+counter[0])
clf =svm.OneClassSVM(nu=impurity,kernel="rbf",gamma=0.005)
clf.fit(x_train.values.tolist())
y_pred1 = clf.predict(x_test.values.tolist())

y_test = y_test['label'].tolist()

check1 = y_pred1==y_test
print ("percentage of samples classified correctly - ",(float(sum(check1))/float(len(check1)))*100)
conf_table(y_pred1,y_test)
# train_df = pd.read_csv("training_NN.csv")
# test_df = pd.read_csv("testing_NN.csv")
