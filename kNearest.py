import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import neighbors, preprocessing
import pandas as pd

df = pd.read_csv('/Users/damilareadedoyin/Documents/Python/DSN/Classification/KNearestNeighbors/breast-cancer-wisconsin.data')
df.replace('?', -9999, inplace= True)

df.drop(['id'], 1, inplace=True)


x = np.array(df.loc[:, df.columns != 'class'])
y = np.array(df['class'])

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
example_measures = np.array([4,2,1,1,3,4,5,6,7])
example_measures = example_measures.reshape(1,-1)
prediction = clf.predict(example_measures)
print(prediction)