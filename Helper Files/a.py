import pandas as pd
import collections
f=pd.read_csv("training_NN.csv",usecols=['label'])
g=pd.read_csv("testing_NN.csv",usecols=['label'])
f=f['label'].tolist()
g=g['label'].tolist()
c1=collections.Counter(f)
c2=collections.Counter(g)
print(c1)
print(c2)
