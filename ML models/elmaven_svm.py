import pandas as pd
from sklearn.svm import SVC
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
print('Reading done...')
##svm = SVC(kernel="linear")
##svm.fit(x_train.values.tolist(),y_train['label'].tolist())
##print('Linear fitting done...')
##y_pred1 = svm.predict(x_test.values.tolist())
##print('Linear prediction done...')
svm2 = SVC(kernel="rbf")
svm2.fit(x_train.values.tolist(),y_train['label'].tolist())
print('RBF fitting done...')
y_pred2 = svm2.predict(x_test.values.tolist())
print('RBF prediction done...')

y_test = y_test['label'].tolist()

##check1 = y_pred1==y_test
##print ("percentage of samples classified correctly - ",(float(sum(check1))/float(len(check1)))*100)
##conf_table(y_pred1,y_test)

check2 = y_pred2==y_test
print ("percentage of samples classified correctly - ",(float(sum(check2))/float(len(check2)))*100)
conf_table(y_pred2,y_test)
print('All done.')
# train_df = pd.read_csv("training_NN.csv")
# test_df = pd.read_csv("testing_NN.csv")
