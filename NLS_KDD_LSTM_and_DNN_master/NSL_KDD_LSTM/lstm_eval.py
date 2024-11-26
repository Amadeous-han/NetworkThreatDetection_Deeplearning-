# -*- coding: utf-8 -*-
import numpy as np
import sklearn as sk
import tensorflow.compat.v1 as tf
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler

from pywebio.input import *
from pywebio.output import *

from NSL_KDD.NLS_KDD_LSTM_and_DNN_master.NSL_KDD_LSTM.lstm_train import load_data

tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
from NSL_KDD.NLS_KDD_LSTM_and_DNN_master.NSL_KDD_LSTM import lstm_inference,lstm_train
'''
def get_test_data(data):
    scaler_for_x = MinMaxScaler(feature_range=(0,1))      
    scaled_x_data = scaler_for_x.fit_transform(data[:,:-1])   
    normalized_test_data = scaled_x_data[:]
    label_test = data[:,-1]
    
    test_x,test_y = [],[]
    size = (len(normalized_test_data)+lstm_train.time_step-1)//lstm_train.time_step
    for i in range(size-1):
        # test_x
        x = normalized_test_data[i*lstm_train.time_step:(i+1)*lstm_train.time_step,:41]
        test_x.append(x.tolist())
        
        # test_y
        y = label_test[i*lstm_train.time_step:(i+1)*lstm_train.time_step]
        labely = []
        for j in y:
            label_list = [0 for num in range(2)]
            label_list[int(j)] = 1
            labely.append(label_list)
        test_y.append(labely)
    return test_x,test_y        
'''

def load_data_lstm(file_path):
    with (open(file_path,'r')) as f:
        df = pd.read_csv(f)
        data = df.iloc[:,:].values
        
    scaler_for_x = MinMaxScaler(feature_range=(0,1))      
    scaled_x_data = scaler_for_x.fit_transform(data[:,:-1]) 
    featurex = scaled_x_data.tolist()    
    
    label_test = data[:,-1]
    labely = []
    for i in label_test:
        label_list = [0 for num in range(2)]
        label_list[int(i)] = 1
        labely.append(label_list)
        
    return featurex,labely
    
def evaluate_lstm(feature,label):
    # input
    X = tf.placeholder(tf.float32,shape=[None,lstm_train.time_step,lstm_inference.input_size],name='x-input')  
    Y = tf.placeholder(tf.float32,shape=[None,lstm_train.time_step,lstm_inference.output_size],name='y-input')  
         
    pred = lstm_inference.lstm(X)  
    Y_ = tf.reshape(Y,[-1,lstm_inference.output_size])
    pred_ = tf.reshape(pred,[-1,lstm_inference.output_size])
    
    # accuracy
    correct_prediction = tf.equal(tf.argmax(pred_, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    # test_x,test_y = get_test_data(data)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(lstm_train.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            
            xs = tf.reshape(feature,[-1,lstm_train.time_step,lstm_inference.input_size])
            ys = tf.reshape(label,[-1,lstm_train.time_step,lstm_inference.output_size])
            accuracy_score = sess.run(accuracy,feed_dict={X:xs.eval(),Y:ys.eval(),lstm_inference.keep_prob:1.0})
            print("step %s,test accuracy is %g" % (step,accuracy_score))
            put_text("step %s,test accuracy is %g" % (step,accuracy_score))
            y_p = tf.argmax(pred_, 1)
            val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={X:xs.eval(),Y:ys.eval(),lstm_inference.keep_prob:1.0})
            print("validation accuracy:", val_accuracy)
            put_text("validation accuracy:", val_accuracy)
            y_true = np.argmax(label, 1)
            threat_index = np.where(y_pred == 0) #0是异常值
            print("Cyber Threat Index",threat_index)
            put_text("Cyber Threat Index:",threat_index)
            put_text("The total number of the Threats:",np.array(threat_index).shape[1])
            print("Precision", sk.metrics.precision_score(y_true, y_pred, average='macro'))
            print("Recall", sk.metrics.recall_score(y_true, y_pred, average='macro'))
            print("f1_score", sk.metrics.f1_score(y_true, y_pred, average='macro'))
            put_text("Precision:", sk.metrics.precision_score(y_true, y_pred, average='macro'))
            put_text("Recall:", sk.metrics.recall_score(y_true, y_pred, average='macro'))
            put_text("f1_score:", sk.metrics.f1_score(y_true, y_pred, average='macro'))
            print("confusion_matrix")

            cm = sk.metrics.confusion_matrix(y_true, y_pred)
            cm_display = ConfusionMatrixDisplay(cm).plot()
            put_text("confusion_matrix:")
            plt.savefig('confusion_matrix_LSTM.jpg')
            plt.show()

            img = open('confusion_matrix_LSTM.jpg', 'rb').read()
            put_image(img, width='1000px')

            put_image('https://www.python.org/static/img/confusion_matrix_LSTM.jpg')
            print(cm)
        else:
            print('no checkpoint file found')
            return

def final(argv=None):
    '''
    file_path = "/home/hy/KDD_MLP/kddtest.csv"
    with (open(file_path,'r')) as f:
        df = pd.read_csv(f)
        data = df.iloc[:,:].values
    print(type(data))
    print("load data success!")
    evaluate(data)
    '''
    file_path = "nslkddtest.csv"
    feature,label = load_data_lstm(file_path)
    print("load data success")
    evaluate_lstm(feature,label)
    
if __name__ == '__main__':
    final()

