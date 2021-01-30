import struct
from sklearn import model_selection, svm, metrics

# def to_csv(name, maxdata):
#     #lable, image 파일 열기 
#     lbl_f = open("./mnist/"+name+"-labels-idx1-ubyte","rb")
#     img_f = open("./mnist/"+name+"-images-idx3-ubyte","rb")
#     csv_f = open("./mnist/"+name+".csv", "w", encoding= "utf-8")
#     #헤더정보 읽기
#     mag, lbl_count = struct.unpack(">II", lbl_f.read(8))  # >는 big-endian, 대문자 i는 unsigned int, 결과는 (2049, 60000). 각각에 저장된다.
#     mag, img_count = struct.unpack(">II", img_f.read(8))  # .read(8)은 8개만 읽겠다는 뜻. 여기서는 buffer이므로, 결과는 (2051, 60000). 각각에 저장된다.

#     rows, cols = struct.unpack(">II", img_f.read(8)) # 8씩 읽어서 각 객체에 저장하게 된다. 결과는 (28, 28). 각각에 저장된다
#     pixels = rows * cols
#     #이미지 데이터를 읽고 저장

#     res = []
#     for idx in range(lbl_count):
#         if idx > maxdata: break
#         label = struct.unpack("B",lbl_f.read(1))[0]
#         bdata = img_f.read(pixels)
#         sdata = list(map(lambda n: str(n), bdata))
#         csv_f.write(str(label) + ",")
#         csv_f.write(",".join(sdata) + "\r\n")

#     csv_f.close()
#     lbl_f.close()
#     img_f.close()

# to_csv("train",99999)
# to_csv("t10k",99999)

def load_csv(fname):
    labels = []
    images = []

    with open(fname, "r") as f:
        for line in f:
            cols = line.split(",")
            if len(cols) < 2 : continue
            labels.append(int(cols.pop(0)))
            vals = list(map(lambda n: int(n) / 256, cols))
            images.append(vals)

    return {"labels":labels, "images":images}

data = load_csv("./mnist/train.csv")
test = load_csv("./mnist/t10k.csv")

#학습

clf = svm.SVC()
clf.fit(data["images"], data["labels"])

#예측
predict = clf.predict(test["images"])

#결과확인
ac_score = metrics.accuracy_score(test["labels"], predict)
cl_report = metrics.classification_report(test["labels"], predict)
print("정답률 : ", ac_score)
print(cl_report)
