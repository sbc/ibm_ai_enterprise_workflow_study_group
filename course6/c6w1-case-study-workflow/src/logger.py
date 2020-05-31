import re
import os
import sys
import time
import csv
import uuid
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
from datetime import date

LOGDIR = "../logs"

def update_predict_log(y_pred,y_proba,query,model_version, runtime, TEST=False):
    """
    update predict log file
    """

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    logfile = "pred-{}-{}.log".format(today.year, today.month)

    if TEST:
        logfile = "test-" + logfile

    logfile = os.path.join(LOGDIR, logfile)

    ## write the data to a csv file    
    header = ['unique_id','timestamp','y_pred','y_proba','x_shape','model_version','runtime']
    write_header = False

    if not os.path.exists(logfile):
        write_header = True

    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(),time.time(),y_pred,y_proba,query.shape,model_version,runtime])
        writer.writerow(to_write)

def update_train_log(data_shape,eval_summary,model_version, runtime, TEST=False):
    """
    update predict log file
    """

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    logfile = "train-{}-{}.log".format(today.year, today.month)

    if TEST:
        logfile = "test-" + logfile

    logfile = os.path.join(LOGDIR, logfile)
    
    ## write the data to a csv file    
    header = ['unique_id','timestamp','data_shape','eval_summary','model_version','runtime']
    write_header = False

    if not os.path.exists(logfile):
        write_header = True

    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(),time.time(),data_shape, eval_summary,model_version,runtime])
        writer.writerow(to_write)



if __name__ == "__main__":

    """
    basic test procedure for model.py
    """
    