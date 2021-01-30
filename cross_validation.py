import pandas as pd
from sklearn import svm, metrics, model_selection
import random, re

csv = pd.read_csv("iris.csv")

data = csv[["SepalLength", "SepalWidth", "PetalLength", "petalWidth"]]
label = csv["Name"]

clf = svm.SVC()
scores = model_selection.cross_val_score(clf, data, label, cv=5)
print(scores)
print("평균", scores.mean())