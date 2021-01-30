from sklearn import svm, metrics
import random, re
from sklearn.model_selection import train_test_split
import pandas as pd
# csv = []
# #csv를 가공할수있는 형태로 바꿔줌
# with open('iris.csv', 'r', encoding='utf-8') as fp:
#     for line in fp:
#         line = line.strip()
#         cols = line.split(',')
#         fn = lambda n : float(n) if re.match('^[0-9\.]+$',n) else n # ^: ~로 시작, \.: '.'이 들어있는가, +: 1개이상 반복, $: ~로 끝 
#         cols = list(map(fn, cols))
#         csv.append(cols)

# del csv[0]

# random.shuffle(csv)

# # 학습 전용 데이터, 테스트 전용 데이터로 나눔

# total_len = len(csv)
# train_len = total_len * (2 / 3) #학습전용 데이터
# train_data = []
# train_label = []
# test_data = []
# test_label = []

# for i in range(total_len):
#     data = csv[i][0:4]
#     label = csv[i][4]
#     if i < train_len:
#         train_data.append(data)
#         train_label.append(label)

#     else:
#         test_data.append(data)
#         test_label.append(label)

# clf = svm.SVC()
# clf.fit(train_data, train_label)
# pre = clf.predict(test_data)
# print(pre)
# print(test_label)
# #정답률 구하기
# ac_score = metrics.accuracy_score(test_label,pre)
# print(ac_score)

#열 추출하기
csv = pd.read_csv('iris.csv')
csv_data = csv[["sepal.length","sepal.width","petal.length","petal.width"]]
csv_label = csv["variety"]

#학슴 전용 데이터와 테스트 전용 데이터 나누기
train_data, test_data, train_label, test_label = train_test_split(csv_data, csv_label, test_size = 0.3)

#데이터 학습 및 예측

clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

#정답률 구하기
ac_score = metrics.accuracy_score(test_label,pre)
print(ac_score)
