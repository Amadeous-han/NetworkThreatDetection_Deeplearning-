import random
from random import randint

oldf = open('KDDTest-21_processed.csv', 'r', encoding='UTF-8')
# newf = open('', 'w', encoding='UTF-8')
test = open('nslkddtest.csv','w', encoding='UTF-8')
train = open('nlskddtrain.csv','w', encoding='UTF-8')
n = 0
# sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素
resultList = random.sample(range(0, 11850), 8001)

lines=oldf.readlines()
for i in resultList:
    train.write(lines[i])

resultList_test = random.sample(range(0, 11850),5001)

lines_test = oldf.readlines()
for i in resultList_test:
    test.write(lines[i])

oldf.close()
train.close()
test.close()