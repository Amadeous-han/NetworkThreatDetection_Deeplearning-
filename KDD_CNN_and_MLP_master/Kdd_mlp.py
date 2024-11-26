#coding:utf-8
from datetime import time
from sklearn import metrics
import sklearn as sk
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random
import csv
global feature
global label
from pywebio.input import *
from pywebio.output import *
#加载训练集(包括特征和标签)
def load_data():
    global feature
    global label
    feature=[]
    label=[]
    filename='kddcup.data_10_percent_corrected.csv'
    with open(filename,'r') as data:
        csv_read=csv.reader(data)
        for row in csv_read:
            feature.append(row[:41])
            label_list=[0 for i in range(23)]
            label_list[int(row[41])]=1
            label.append(label_list)

#Function for adding layers to a neural network
def addLayer(inputs,in_size,out_size,activationFunction=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]),name='Weights')
    Biases=tf.Variable(tf.zeros([1,out_size])+0.1,name='biases')
    W_plus_b=tf.add(tf.matmul(inputs,Weights),Biases)
    if activationFunction is None:
        outputs=W_plus_b
    else:
        outputs=activationFunction(W_plus_b)
    return outputs


#Defining batch gradient descent
def next_batch(feature_list,label_list,size):
    feature_batch_temp=[]
    label_batch_temp=[]
    f_list=random.sample(range(len(feature_list)),size)
    for i in f_list:
        feature_batch_temp.append(feature_list[i])
        label_batch_temp.append(label_list[i])
    return feature_batch_temp,label_batch_temp

from sklearn.datasets import load_iris
import pandas as pd

def kdd_main():
    global feature
    global label
    load_data()  #加载数据
    # 定义训练集
    feature_train=feature[:485000]
    label_train=label[:485000]
    #定义测试集
    feature_test=feature[484000:]
    label_test=label[484000:]

    #Set up placeholder
    picData=tf.placeholder(tf.float32,[None,41],name='picData')
    picLabel=tf.placeholder(tf.float32,[None,23],name='picLabel')

    #add neural network layers
    picPrediction=addLayer(picData,41,23,activationFunction=tf.nn.softmax)

    #Calculating the loss function
    loss=-tf.reduce_sum(picLabel*tf.log(picPrediction),name='loss')

    #Define learning rate and optimise using gradient descent algorithm
    lr = tf.Variable(0.01, dtype=tf.float32)
    trainStep=tf.train.GradientDescentOptimizer(lr,name='trainStep').minimize(loss)

    #Evaluating the effectiveness of test models
    correct_prediction=tf.equal(tf.argmax(picPrediction,1),tf.argmax(picLabel,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #创建会话
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) #初始化变量
        import time
        start = time.time()
        #进行训练1001次
        for step in range(1001):
            #随机批量梯度下降训练，每次选大小为1000的batch
            feature_train_batch,label_train_batch=next_batch(feature_train,label_train,1000)
            sess.run(trainStep,feed_dict={picData:feature_train_batch,picLabel:label_train_batch})

            if step%50==0:
                #测试数据的准确率
                test_accuracy=sess.run(accuracy,feed_dict={picData:feature_test,picLabel:label_test})
                #训练数据的准确率
                train_accuracy=sess.run(accuracy,feed_dict={picData:feature_train_batch,picLabel:label_train_batch})
                print(step,test_accuracy,train_accuracy)
                # put_text("step:",step,"test_accuracy:",test_accuracy,"train_accuracy:",train_accuracy)
         # metric
        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        y_p = tf.argmax(picPrediction, 1)
        val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={picData: feature_test, picLabel: label_test})
        print("validation accuracy:", val_accuracy)
        put_text("validation accuracy:", val_accuracy)
        y_true = np.argmax(label_test, 1)
        threat_index = np.where(y_pred != 1)  #
        print("Cyber Threat Index", threat_index)
        put_text("Cyber Threat Index:", threat_index)
        put_text("The total number of the Threats:", np.array(threat_index).shape[1])
        print("Report")
        print(classification_report(y_true, y_pred, digits=3))
        cm=sk.metrics.confusion_matrix(y_true, y_pred)
        print("Precision:", sk.metrics.precision_score(y_true, y_pred,average='macro'))
        print("Recall:", sk.metrics.recall_score(y_true, y_pred,average='macro'))
        print("f1_score:", sk.metrics.f1_score(y_true, y_pred,average='macro'))
        put_text("Precision:", sk.metrics.precision_score(y_true, y_pred,average='macro'))
        put_text("Recall:", sk.metrics.recall_score(y_true, y_pred,average='macro'))
        put_text("f1_score:", sk.metrics.f1_score(y_true, y_pred,average='macro'))
        print("confusion_matrix")
        cm=sk.metrics.confusion_matrix(y_true, y_pred)
        cm_display = ConfusionMatrixDisplay(cm).plot()
        put_text("confusion_matrix:")
        plt.savefig('confusion_matrix_MLP.jpg')
        plt.show()

        img = open('confusion_matrix_MLP.jpg', 'rb').read()
        put_image(img, width='1000px')

        put_image('https://www.python.org/static/img/confusion_matrix_MLP.jpg')
        print(cm)

        fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.title('Validation ROC')
        plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()







if __name__=='__main__':
    global feature
    global label
    kdd_main()

