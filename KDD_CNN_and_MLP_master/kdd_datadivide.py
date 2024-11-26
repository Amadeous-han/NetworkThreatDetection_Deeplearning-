import random
from random import randint

oldf = open('kddcup.data_10_percent_corrected.csv', 'r', encoding='UTF-8')
# newf = open('', 'w', encoding='UTF-8')
test = open('kddtest.csv','w', encoding='UTF-8')
train = open('kddtrain.csv','w', encoding='UTF-8')
n = 0
# sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素
resultList = random.sample(range(0, 494021), 400000)

lines=oldf.readlines()
for i in resultList:
    train.write(lines[i])

resultList_test = random.sample(range(0, 494021), 20000)

lines_test = oldf.readlines()
for i in resultList_test:
    test.write(lines[i])

oldf.close()
train.close()
test.close()