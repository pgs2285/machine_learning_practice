from sklearn import svm, metrics
import glob, os.path, re, json

def check_freq(fname): # 텍스트를 읽고, 빈도조사
    name = os.path.basename(fname)
    lang = re.match(r'^[a-z]{2,}', name).group() #en-1.txt 면 앞에 en만 따오기
    with open(fname, "r", encoding="utf-8") as f:
        text = f.read()
    text = text.lower()
    cnt = [0 for n in range(0,26)]
    code_a = ord("a") # 아스키 코드값으로 변환해줌
    code_z = ord("z")

    for ch in text:    
        n = ord(ch) 
        if code_a <= n <= code_z: #a~z 사이
            cnt[n - code_a] += 1 # 알파벳 빈도수 측정 0~25 순으로 a~z
    
    total = sum(cnt)
    freq = list(map(lambda n: n / total, cnt)) #정규화 시키기    
    return (freq, lang)

def load_files(path): #파일 처리하기
    freqs = []
    labels = []
    file_list = glob.glob(path)
    for fname in file_list:
        r = check_freq(fname)
        freqs.append(r[0])
        labels.append(r[1])
    return {"freqs":freqs, "labels": labels} 

data = load_files("./lang/train/*.txt") #경로의 txt파일
test = load_files("./lang/test/*.txt")

with open("./lang/freq.json","w", encoding = "utf-8") as fp: #json 파일로 저장, json은 "키-값 쌍"으로 이루어진 데이터 오브젝트를 전달하기위해 텍스트를 사용하는 개방형 표준 포맷 
    json.dump([data, test], fp) #dump는 python을 json으로 쓰는것

clf = svm.SVC()
clf.fit(data["freqs"], data["labels"])


predict = clf.predict(test["freqs"]) #예측

ac_score = metrics.accuracy_score(test["labels"], predict)
cl_report = metrics.classification_report(test["labels"],predict)
print("정답률 :", ac_score)
print("리포트 :")
print(cl_report)


