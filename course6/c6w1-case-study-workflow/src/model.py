import time,os,re,csv,sys,uuid,joblib
from datetime import date
import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from logger import update_predict_log, update_train_log

## model specific variables (iterate the version and note with each change)
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "example random forest on toy data"
MODEL_DIR = "../models"
SAVED_MODEL = os.path.join(MODEL_DIR, "model-{}.joblib".format(re.sub("\.","_",str(MODEL_VERSION))))

def fetch_data():
    """
    example function to fetch data for training
    """
    
    ## import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:,:2]
    y = iris.target

    return(X,y)
    
def model_train(mode=None, test=False):
    """
    example funtion to train model
    
    'mode' -  can be used to subset data essentially simulating a train
    """

    ## start timer for runtime
    time_start = time.time()

    ## data ingestion
    X,y = fetch_data()

    ## Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    ## Specify parameters and model
    params = {'C':1.0,'kernel':'linear','gamma':0.5}
    clf = svm.SVC(**params,probability=True)

    ## fit model on training data
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test,y_pred))

    ## retrain using all data
    # clf.fit(X, y)
    print("... saving model: {}".format(SAVED_MODEL))
    joblib.dump(clf,SAVED_MODEL)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    eval_summary = {'rmse': mean_squared_error(y_test, y_pred, squared=False)}

    update_train_log(X.shape,eval_summary,MODEL_VERSION, runtime, test)

def model_load():
    """
    example funtion to load model
    """

    if not os.path.exists(SAVED_MODEL):
        raise Exception("Model '{}' cannot be found did you train the model?".format(SAVED_MODEL))
    
    model = joblib.load(SAVED_MODEL)
    return(model)

def model_predict(query,model=None,test=False):
    """
    example funtion to predict from model
    """

    ## start timer for runtime
    time_start = time.time()

    ## load model if needed
    if not model:
        model = model_load()
    
    ## output checking
    if len(query.shape) == 1:
        query = query.reshape(1, -1)
    
    ## make prediction and gather data for log entry
    y_pred = model.predict(query)
    y_proba = None
    if 'predict_proba' in dir(model) and model.probability == True:
        y_proba = model.predict_proba(query)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    update_predict_log(y_pred,y_proba,query,MODEL_VERSION,runtime,test)
        
    return({'y_pred':y_pred,'y_proba':y_proba})

if __name__ == "__main__":

    """
    basic test procedure for model.py
    """
    
    ## train the model
    model_train()

    ## load the model
    model = model_load()
    
    ## example predict
    for query in [np.array([[6.1,2.8]]), np.array([[7.7,2.5]]), np.array([[5.8,3.8]])]:
        result = model_predict(query,model)
        y_pred = result['y_pred']
        print("predicted: {}".format(y_pred))



