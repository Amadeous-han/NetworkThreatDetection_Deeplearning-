import sys

from pywebio.input import *
from pywebio.output import *

from KDD99.KDD_LSTM_and_DNN_master.KDD_DNN.dnn_eval import *
from KDD99.KDD_LSTM_and_DNN_master.KDD_LSTM.lstm_eval import *
from KDD99.KDD_LSTM_and_DNN_master.KDD_DNN.dnn_train import *
from KDD99.KDD_LSTM_and_DNN_master.KDD_LSTM.lstm_train import *
from KDD99.KDD_LSTM_and_DNN_master.KDD_DNN.dnn_inference import *
from KDD99.KDD_LSTM_and_DNN_master.KDD_LSTM.lstm_inference import *
from KDD_CNN_and_MLP_master.kdd_cnn import *
from KDD_CNN_and_MLP_master.Kdd_mlp import *
from KDD_LSTM_and_DNN_master.KDD_DNN import *
from KDD_LSTM_and_DNN_master.KDD_LSTM import *
from KDD_LSTM_and_DNN_master.KDD_DNN import dnn_inference
if __name__=="__main__":
    put_markdown(r""" # Detection Platform
        This is the platform show all the results.""", strip_indent=4).show()
    while (1):
        confirm = actions('Do you want to exit?', ['YES Exit', 'NO Go on'],
                      help_text='Unrecoverable after file deletion')
        if confirm == 'NO Go on':
    # database = select("which database do you want to use",["KDD 99"," "," "])
            model = select("Which model do you want to chosoe",["","MLP","DNN","CNN","LSTM"])
            if model == "MLP":
                clear()
                put_markdown(r""" # MLP Model
                        You are now in MLP model""", strip_indent=4).show()
                database = select("which database do you want to use", ["KDD 99", " ", " "])
                # kdd_main()
            if model == "DNN":
                clear()
                put_markdown(r""" # DNN Model
                                        You are now in DNN model""", strip_indent=4).show()
                database = select("which database do you want to use", ["KDD 99", " ", " "])
                file_path = "../kddtest.csv"
                feature, label = load_data_dnn(file_path)
                evaluate_dnn(feature, label)
            if model == "CNN":
                clear()
                put_markdown(r""" # CNN Model
                    You are now in CNN model""", strip_indent=4).show()
                database = select("which database do you want to use", ["KDD 99", " ", " "])
                # cnn_main()
            if model == "LSTM":
                clear()
                put_markdown(r""" # LSTM Model
                You are now in LSTM model""", strip_indent=4).show()
                database = select("which database do you want to use", ["KDD 99", " ", " "])
                file_path = "../kddtest_lstm.csv"
                feature, label = load_data_lstm(file_path)
                evaluate_lstm(feature, label)
        if confirm == 'YES Exit':
            clear()
            sys.exit(0)