# import random

# def calc_bmi(h, w):
#     bmi = w/ (h/100) ** 2 # bmi를 구하는 공식
#     if bmi < 18.5: return "저체중"
#     if bmi <25: return "정상체중"
#     return "비만"

# fp = open("bmi.csv", "w", encoding = "utf-8")
# fp.write("height,weight,label\r\n")    

# cnt = {"저체중": 0, "정상체중":0, "비만":0}
# for i in range(20000):
#     h = random.randint(120,200)
#     w = random.randint(35, 80)
#     label = calc_bmi(h, w)
#     cnt[label] +=1
#     fp.write("{0},{1},{2}\r\n".format(h, w, label))
# fp.close()
# print("ok,",cnt)    
# csv데이터 임의생성


# from sklearn import svm, metrics
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import pandas as pd

# tbl = pd.read_csv("bmi.csv")

# label = tbl["label"]
# w = tbl["weight"]/80 #최대 80kg 정규화
# h = tbl["height"]/200 #최대 2m 정규화
# wh = pd.concat([w, h], axis = 1) #concat 은 데이터 합치기 , axis = 1 은 왼쪽 + 오른쪽 합치기 

# data_train, data_test, label_train, label_test = train_test_split(wh, label)

# clf = svm.SVC()
# clf.fit(data_train,label_train)

# predict = clf.predict(data_test)

# ac_score = metrics.accuracy_score(label_test, predict)
# cl_report = metrics.classification_report(label_test, predict)

# print("정답률 = ", ac_score)
# print("리포트 = \n", cl_report)

import matplotlib.pyplot as plt
import pandas as pd

tbl = pd.read_csv("bmi.csv", index_col = 2) # label만 가져오기
fig = plt.figure()#새로운 figure 생성
ax = fig.add_subplot(2, 2, 2)

def scatter(lbl, color):
    b = tbl.loc[lbl] #loc 는 데이터 추출(index기준)
    if lbl == "비만": lbl = "fat"
    elif lbl == "정상체중": lbl = "normal"
    else: lbl = "thin" 
    ax.scatter(b["weight"], b["height"], c=color, label = lbl)

scatter("비만", "red")
scatter("정상체중", "yellow")
scatter("저체중", "purple")

ax.legend()
plt.savefig("bmi-test.png")

