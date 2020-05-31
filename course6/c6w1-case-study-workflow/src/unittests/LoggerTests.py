"""
logger tests
"""
import os
import unittest
from datetime import date
import numpy as np
import pandas as pd
import csv
import requests

import sys
sys.path.insert(1, os.path.realpath(os.path.pardir)) 
# print(sys.path)

MODEL_VERSION = "2.3"
LOGDIR = "../logs"
HOST = "localhost"
PORT = "8080"

## import model specific functions and variables
from src.logger import update_predict_log, update_train_log

def the_testlogname(mode):
    today = date.today()
    logfile = "test-{}-{}-{}.log".format(mode, today.year, today.month)
    logfile = os.path.join(LOGDIR, logfile)

    return logfile

class LoggerTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_create_pred_log(self):
        """
        test that predict log is created
        """

        logfile = the_testlogname("pred")
        
        if os.path.exists(logfile):
            os.remove(logfile)

        update_predict_log([0],[0,0,0],np.array([6.1, 2.8]),MODEL_VERSION,0.05, True)
        
        self.assertTrue(os.path.exists(logfile))

    def test_02_create_train_log(self):
        """
        test that train log is created
        """

        logfile = the_testlogname("train")

        if os.path.exists(logfile):
            os.remove(logfile)

        # update_train_log(data_shape,eval_summary,model_version, runtime)
        update_train_log( (150,2),{'rmse':0.8},"9.9",0., True)

        self.assertTrue(os.path.exists(logfile))

    def test_03_archive_train_data(self):

        """
        test that train data is logged
        """

        logfile = the_testlogname("train")

        if os.path.exists(logfile):
            os.remove(logfile)

        # update_train_log(data_shape,eval_summary,model_version, runtime)
        data_shape = (150,2)
        eval_summary = {'rmse':0.8}
        model_version = MODEL_VERSION
        runtime = 0.05
        update_train_log(data_shape, eval_summary, model_version, runtime, True)

        # get last row of log
        df = pd.read_csv(logfile, delimiter=',', quotechar='|')
        last = df.tail(1).iloc[0].to_dict()
        
        self.assertEqual(last['eval_summary'], str(eval_summary))
        self.assertEqual(last['data_shape'], str(data_shape))

    def test_04_check_n_preds(self):

        """
        test n predictions add n log lines
        """
        n = 5

        logfile = the_testlogname("pred")

        if not os.path.exists(logfile):
            before = 0
        else: 
            df = pd.read_csv(logfile, delimiter=',', quotechar='|')
            before = df.shape[0]

        for i in range(n):
            update_predict_log([0],[0,0,0],np.array([6.1, 2.8]),MODEL_VERSION,0.05, True)

        df = pd.read_csv(logfile, delimiter=',', quotechar='|')
        after = df.shape[0]
        
        self.assertEqual(n, after-before)

    def test_05_check_pred_nans(self):

        pass


### Run the tests
if __name__ == '__main__':
    unittest.main()
