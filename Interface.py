import sys

import tf as tf
from pywebio.input import *
from pywebio.output import *

import KDD99.KDD_LSTM_and_DNN_master.KDD_LSTM.lstm_eval
import NSL_KDD.NSL_KDD_CNN_and_MLP_master.nlskdd_mlp
import UNSW_NB15.UNSW_NB15_CNN_and_MLP_master.unswnb15_mlp
from KDD99.KDD_LSTM_and_DNN_master.KDD_DNN.dnn_eval import *
from KDD99.KDD_LSTM_and_DNN_master.KDD_LSTM.lstm_eval import *
from KDD99.KDD_LSTM_and_DNN_master.KDD_DNN.dnn_train import *
from KDD99.KDD_LSTM_and_DNN_master.KDD_LSTM.lstm_train import *
from KDD99.KDD_LSTM_and_DNN_master.KDD_DNN.dnn_inference import *
from KDD99.KDD_LSTM_and_DNN_master.KDD_LSTM.lstm_inference import *
from KDD99.KDD_CNN_and_MLP_master.kdd_cnn import *
from KDD99.KDD_CNN_and_MLP_master.Kdd_mlp import *
from KDD99.KDD_LSTM_and_DNN_master.KDD_DNN import *
from KDD99.KDD_LSTM_and_DNN_master.KDD_LSTM import *
from KDD99.KDD_LSTM_and_DNN_master.KDD_DNN import dnn_inference
from NSL_KDD.NSL_KDD_CNN_and_MLP_master.nlskdd_cnn import *
from NSL_KDD.NSL_KDD_CNN_and_MLP_master.nlskdd_mlp import *
from NSL_KDD.NLS_KDD_LSTM_and_DNN_master.NSL_KDD_DNN import *
import NSL_KDD.NLS_KDD_LSTM_and_DNN_master.NSL_KDD_DNN.dnn_eval
import NSL_KDD.NLS_KDD_LSTM_and_DNN_master.NSL_KDD_LSTM.lstm_eval
import NSL_KDD.NLS_KDD_LSTM_and_DNN_master.NSL_KDD_LSTM.lstm_train
import UNSW_NB15.UNSW_NB15_LSTM_and_DNN_master.UNSW_NB15_DNN.dnn_eval
import UNSW_NB15.UNSW_NB15_LSTM_and_DNN_master.UNSW_NB15_LSTM.lstm_train
import UNSW_NB15.UNSW_NB15_LSTM_and_DNN_master.UNSW_NB15_LSTM.lstm_eval
from NSL_KDD.NLS_KDD_LSTM_and_DNN_master.NSL_KDD_LSTM import *
from UNSW_NB15.UNSW_NB15_CNN_and_MLP_master.unswnb15_cnn import *
from UNSW_NB15.UNSW_NB15_CNN_and_MLP_master.unswnb15_mlp import *
if __name__=="__main__":
    put_markdown(r""" # Detection Platform
        This is the platform show all the results.""", strip_indent=4).show()
    while (1):
        confirm = actions('Do you want to exit?', ['YES Exit', 'NO Go on'],
                      help_text='Unrecoverable after file deletion')
        if confirm == 'NO Go on':
            # database = select("which database do you want to use",["KDD 99","NSL_KDD","UNSW_NB15"])
            model = select("Which model do you want to chosoe",["","MLP","DNN","CNN","LSTM"])
            if model == "MLP":
                clear()
                put_markdown(r""" # MLP Model
                        You are now in MLP model""", strip_indent=4).show()
                database = select("which database do you want to use", ["","KDD 99","NSL_KDD","UNSW_NB15"])
                if database == "KDD 99":
                    put_text("you are using KDD 99")
                    KDD99.KDD_CNN_and_MLP_master.Kdd_mlp.kdd_main()
                if database == "NSL_KDD":
                    put_text("you are using NSL_KDD")
                    NSL_KDD.NSL_KDD_CNN_and_MLP_master.nlskdd_mlp.kdd_main()
                if database == "UNSW_NB15":
                    put_text('you are using UNSW_NB15')
                    UNSW_NB15.UNSW_NB15_CNN_and_MLP_master.unswnb15_mlp.kdd_main()
            if model == "DNN":
                clear()
                put_markdown(r""" # DNN Model
                                        You are now in DNN model""", strip_indent=4).show()
                database = select("which database do you want to use", ["","KDD 99","NSL_KDD","UNSW_NB15"])
                if database == "KDD 99":
                    put_text("you are using KDD 99")
                    file_path = "kddtest.csv"
                    feature, label = KDD99.KDD_LSTM_and_DNN_master.KDD_DNN.dnn_eval.load_data_dnn(file_path)
                    KDD99.KDD_LSTM_and_DNN_master.KDD_DNN.dnn_eval.evaluate_dnn(feature, label)
                if database == "NSL_KDD":
                    put_text("you are using NSL_KDD")
                    file_path = "nslkddtest.csv"
                    feature, label = NSL_KDD.NLS_KDD_LSTM_and_DNN_master.NSL_KDD_DNN.dnn_eval.load_data_dnn(file_path)
                    NSL_KDD.NLS_KDD_LSTM_and_DNN_master.NSL_KDD_DNN.dnn_eval.evaluate_dnn(feature, label)
                if database == "UNSW_NB15":
                    put_text('you are using UNSW_NB15')
                    file_path = "nb15test.csv"
                    feature, label = UNSW_NB15.UNSW_NB15_LSTM_and_DNN_master.UNSW_NB15_DNN.dnn_eval.load_data_dnn(file_path)
                    UNSW_NB15.UNSW_NB15_LSTM_and_DNN_master.UNSW_NB15_DNN.dnn_eval.evaluate_dnn(feature, label)

            if model == "CNN":
                clear()
                put_markdown(r""" # CNN Model
                    You are now in CNN model""", strip_indent=4).show()
                database = select("which database do you want to use", ["","KDD 99","NSL_KDD","UNSW_NB15"])
                if database == "KDD 99":
                    put_text("you are using KDD 99")
                    KDD99.KDD_CNN_and_MLP_master.kdd_cnn.cnn_main()
                if database == "NSL_KDD":
                    put_text("you are using NSL_KDD")
                    NSL_KDD.NSL_KDD_CNN_and_MLP_master.nlskdd_cnn.cnn_main()
                if database == "UNSW_NB15":
                    put_text('you are using UNSW_NB15')
                    UNSW_NB15.UNSW_NB15_CNN_and_MLP_master.unswnb15_cnn.cnn_main()
            if model == "LSTM":
                clear()
                put_markdown(r""" # LSTM Model
                You are now in LSTM model""", strip_indent=4).show()
                database = select("which database do you want to use", ["","KDD 99","NSL_KDD","UNSW_NB15"])
                if database == "KDD 99":
                    put_text("you are using KDD 99")
                    file_path = "kddtest_lstm.csv"
                    feature, label = KDD99.KDD_LSTM_and_DNN_master.KDD_LSTM.lstm_eval.load_data_lstm(file_path)
                    KDD99.KDD_LSTM_and_DNN_master.KDD_LSTM.lstm_eval.evaluate_lstm(feature, label)
                if database == "NSL_KDD":
                    put_text("you are using NSL_KDD")
                    file_path = "nslkddtest.csv"
                    feature, label = NSL_KDD.NLS_KDD_LSTM_and_DNN_master.NSL_KDD_LSTM.lstm_eval.load_data_lstm(file_path)
                    # NSL_KDD.NLS_KDD_LSTM_and_DNN_master.NSL_KDD_LSTM.lstm_eval.final()
                    NSL_KDD.NLS_KDD_LSTM_and_DNN_master.NSL_KDD_LSTM.lstm_train.train_lstm()
                if database == "UNSW_NB15":
                    put_text('you are using UNSW_NB15')
                    file_path = "nb15test.csv"
                    feature, label = UNSW_NB15.UNSW_NB15_LSTM_and_DNN_master.UNSW_NB15_LSTM.lstm_eval.load_data_lstm(
                        file_path)
                    UNSW_NB15.UNSW_NB15_LSTM_and_DNN_master.UNSW_NB15_LSTM.lstm_train.train_lstm()

        if confirm == 'YES Exit':
            clear()
            sys.exit(0)