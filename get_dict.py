from gensim import models
import pickle
import numpy as np
from numpy import float32

print("loading model...")

w = models.KeyedVectors.load_word2vec_format('ch.bin', binary=True)

print("done")

dict1 = {}
dict2 = {}

data1 = open("data/cpbtrain.txt").read().splitlines()
data2 = open("data/cpbdev.txt").read().splitlines()
data3 = open("data/cpbtest.txt").read().splitlines()
data = data1 + data2
print(len(data))
data = data + data3
lst = list(map(lambda x: x.split(' ')[:-1], data))

lst2 = list(map(lambda x: list(map(lambda y: y.split('/'), x)), lst))

i = 1
j = 1

for v1 in lst2:
    for v2 in v1:
        if not (v2[1] in dict1):
            dict1[v2[1]] = i
            i+=1
        if len(v2) == 3 and (not (v2[2] in dict2)):
            dict2[v2[2]] = j
            j+=1
i = i-1
j = j-1
print(i)
print(j)

rel = 2

for key, v in dict1.items():
    tmp = [0]*i
    tmp[v-1] = 1
    dict1[key] = tmp

punc = "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."

ex = {}
tmp = 0.1
def word2vec(str):
    global tmp
    if str in w:
        return w[str].tolist()
    elif str in ex:
        return ex[str]
    elif str in punc:
        return [0.0]*64
    else:
        tmp += 0.01
        ex[str] = [tmp]*64
        return ex[str]
        
lst3 = list(map(lambda x: list(map(lambda y: [word2vec(y[0]), dict1[y[1]], (dict2[y[2]] if len(y)== 3 else 0) ] , x)), lst2))

input = []
x = [0.0]*64
y = [0]*32
for v1 in lst3:
    t = 0
    c = []
    p = []

    for i in range(len(v1)):
        if len(v1) == 3 and v1[i][2] == rel:
            t = i
            break
    t -= 2
    for j in range(5):
        if (j + t) < 0 or (j+t) >= len(v1):
            c.append(x)
            p.append(y)
        else:
            c.append(v1[t+j][0])
            p.append(v1[t+j][1])
    t += 2

    d1 = []
    for i in range(len(v1)):

        k = i - 2

        tmp = []
        tmp.append(v1[i][0])
        tmp.append(v1[t][0])
        tmp = tmp + c + p
        if i < t - 15 or i > t + 15:
            tmp.append([0]*31)
        else:
            j = i - t + 15
            v = [0]*31
            v[j] = 1
            tmp.append(v)
        tmp.append(v1[i][1])
        if len(v1[i]) == 3:
            tmp.append(v1[i][2])
        else:
            tmp.append(0)        
        d1.append(tmp)
    input.append(d1)

input = list(filter(lambda x: not x == [], input))

f = open('data/train.data', 'wb')
pickle.dump(input, f, protocol=2)
f.close()
f = open('data/dict.data', 'wb')
pickle.dump(dict2, f)
f.close()