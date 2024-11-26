# -*- coding: utf-8 -*-
import numpy as np
import sklearn as sk
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow.compat.v1 as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from NSL_KDD.NLS_KDD_LSTM_and_DNN_master.NSL_KDD_DNN import dnn_inference, dnn_train


from pywebio.input import *
from pywebio.output import *

'''
def load_data():
    featurex = []
    labely = []
    file_path = "/home/hy/KDD_MLP/kddtest.csv"
    with (open(file_path,'r')) as data_from:
        csv_reader = csv.reader(data_from)
        for j in csv_reader:
            temp = [float(n) for n in j]
            featurex.append(temp[:41])
            label_list = [0 for num in range(2)]
            label_list[int(j[41])] = 1
            labely.append(label_list)
    return featurex,labely
'''

def load_data_dnn(file_path):
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
    
def evaluate_dnn(feature,label):
    with tf.Graph().as_default() as g:
        # input
        x = tf.placeholder(tf.float32,[None,dnn_inference.INPUT_NODE],name='x-input')
        y_ = tf.placeholder(tf.float32,[None,dnn_inference.OUTPUT_NODE],name='y-input')
        validate_feed = {x:feature,y_:label}
        
        # accuracy
        y = dnn_inference.inference(x)
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
        variable_averages = tf.train.ExponentialMovingAverage(dnn_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(dnn_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
                print("step: %s ,test accuracy: %g" % (global_step,accuracy_score))
                put_text("step: %s ,test accuracy: %g" % (global_step,accuracy_score))
            else:
                print('no checkpoint file found')
                return
            # metric
            y_p = tf.argmax(y, 1)
            val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict=validate_feed)
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
            plt.savefig('confusion_matrix_eval.jpg')
            plt.show()

            img = open('confusion_matrix_eval.jpg', 'rb').read()
            put_image(img, width='1000px')
            put_image('https://www.python.org/static/img/confusion_matrix_eval.jpg')
            print(cm)
def main(argv=None):
    file_path = "nslkddtest.csv"
    feature,label = load_data_dnn(file_path)
    print("load data success")
    evaluate_dnn(feature,label)
    
if __name__ == '__main__':
    tf.app.run()
        
