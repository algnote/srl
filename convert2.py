str1 = open("data/cpbdev.txt").read().splitlines()
str2 = open("data/cpbtest.txt").read().splitlines()+['']

str3 = open("data/cpbdev_answer1.txt").read().splitlines()
str4 = open("data/cpbtest_answer1.txt").read().splitlines()+['']

for i in range(len(str1)):
    s1 = str1[i].split(' ')
    s3 = str3[i].split(' ')+['']
    for j in range(len(s1)):
        a = s1[j].split('/')
        b = s3[j].split('/')
        if len(a) == 3:
            a[2] = b[1]
        s1[j] = '/'.join(a)
    str1[i] = ' '.join(s1)

for i in range(len(str2)):
    s1 = str2[i].split(' ')
    s3 = str4[i].split(' ')+['']
    for j in range(len(s1)):
        a = s1[j].split('/')
        b = s3[j].split('/')
        if len(a) == 2:
            a.append(b[1])
        s1[j] = '/'.join(a)
    str2[i] = ' '.join(s1)


f = open ( 'dev.txt', 'w')
for line in str1:
    f.write(line)
    f.write('\n')
f.close()

f = open ( 'test.txt', 'w')
for line in str2:
    f.write(line)
    f.write('\n')
f.close()