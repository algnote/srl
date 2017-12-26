import pickle

str1 = open("data/cpbdev.txt").read().splitlines()
str2 = open("data/cpbtest.txt").read().splitlines()

f = open('data/dev.pre', 'rb')
dev = pickle.load(f)

f = open('data/test.pre', 'rb')
test = pickle.load(f)

f = open('data/dict.data', 'rb')
dic = pickle.load(f)

rdic = {}

for k, v in dic.items():
    rdic[v] = k

i = 0
for k in range(len(str1)):
    s = str1[k].split(' ')
    for j in range(len(s)):
        if i == len(dev):
            break
        a = s[j].split('/')
        if len(a) == 3:
            a[2] = rdic[dev[i]]
            i += 1
        s[j] = "/".join(a)
    b = " ".join(s)
    str1[k] = b

i = 0
for k in range(len(str2)):
    s = str2[k].split(' ')
    for j in range(len(s)):
        if i == len(test):
            break
        a = s[j].split('/')
        if len(a) == 2:
            a.append(rdic[test[i]])
            i += 1
        s[j] = "/".join(a)
    b = " ".join(s)
    str2[k] = b

f = open ( 'dev1.txt', 'w')
for line in str1:
    f.write(line)
    f.write('\n')
f.close()

f = open ( 'test1.txt', 'w')
for line in str2:
    f.write(line)
    f.write('\n')
f.close()