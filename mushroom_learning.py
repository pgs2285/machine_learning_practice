# import urllib.request as req

# local= "mushroom.csv"

# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

# req.urlretrieve(url, local)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

mr = pd.read_csv("mushroom.csv", header = None)

label = []
data = []
attr_list = []

for row_index, row in mr.iterrows():
    label.append(row[0]) #독이 있나 없나 구분하기 위해 독야부가 들은 첫번째를 때서 출력, ix = loc

    row_data = []
    for v in row[1:]:
        row_data.append(ord(v))
    data.append(row_data)    

data_train, data_test, label_train, label_test = train_test_split(data, label)

clf = RandomForestClassifier()
clf.fit(data_train, label_train)

predict = clf.predict(data_test)

ac_score = metrics.accuracy_score(label_test, predict)
cl_report = metrics.classification_report(label_test,predict)

print("정답률:", ac_score)
print("리포트 \n", cl_report)