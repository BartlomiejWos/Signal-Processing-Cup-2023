from . import dataset


from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt


def load_result(nr, path=Path("../notebooks")):

    num = str(nr)
    if len(num) == 1:
        num = "0"+num

    train_path = path / ( num + "_train.csv")
    test_path = path / ( num + "_test.csv")

    train = pd.read_csv(str(train_path), sep=',').Predicted.to_numpy()
    test = pd.read_csv(str(test_path), sep=',').Predicted.to_numpy()
    
    BP_tr=0
    SH_tr=0
    BP_ts=0
    SH_ts=0
    
    BP_tr_ind = []
    SH_tr_ind = []
    BP_ts_ind = []
    SH_ts_ind = []
    for i, tr in enumerate(train):
        if(tr >= 0.5):
            BP_tr = BP_tr + 1
            BP_tr_ind.append(i)
        else:
            SH_tr = SH_tr + 1
            SH_tr_ind.append(i)
            
    for i, ts in enumerate(test):
        if(ts >= 0.5):
            BP_ts = BP_ts + 1
            BP_ts_ind.append(i)
        else:
            SH_ts = SH_ts + 1
            SH_ts_ind.append(i)
            



    # Create a list of y-values for the first vector
    y1 = [1] * len(BP_ts_ind)

    # Create a list of y-values for the second vector
    y2 = [-1] * len(SH_ts_ind)

    # Plot the vectors
    plt.figure(figsize=(27, 1))
    
    plt.scatter(BP_ts_ind, y1, marker='+', label="BP", color='r')
    plt.scatter(SH_ts_ind, y2, marker='x', label="SH", color='b')

    # Show the plot
    plt.legend()
    plt.show()

    return train, test, BP_tr, SH_tr, BP_ts, SH_ts

def show_dist(label_train, train_pred, test_pred, label_valid=None, valid_pred=None):
    bins = np.linspace(0, 1, 100)
    


    AUC = roc_auc_score(label_train, train_pred)
    fpr, tpr, ths = roc_curve(label_train, train_pred)

    if label_valid == None and valid_pred == None:
        fig, (axtrain_dist, axtrain_roc, axtest_dist) = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    else:
        fig, (axtrain_dist, axtrain_roc, axvalid_dist, axvalid_roc, axtest_dist) = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))

    axtrain_dist.set_title("Probability distribution on train")
    axtrain_dist.hist(train_pred, bins, label="SZ+BP", color='b')
    axtrain_dist.hist(train_pred[label_train == 1.0], bins, alpha=0.75, label="BP = 1", color='r')
    axtrain_dist.hist(train_pred[label_train == 0.0], bins, alpha=0.75, label="SZ = 0", color='g')
    axtrain_dist.legend()

    axtrain_roc.set_title("ROC on train")
    axtrain_roc.plot(fpr, tpr, label="AUC = "+str(AUC)[:6])
    axtrain_roc.legend()

    if label_valid != None and valid_pred != None:
        AUC = roc_auc_score(label_valid, valid_pred)
        fpr, tpr, ths = roc_curve(label_valid, valid_pred)
        axtrain_roc.set_title("Probability distribution on valid")
        axtrain_roc.hist(valid_pred, bins, label="SZ+BP")
        axtrain_roc.hist(valid_pred[label_valid == 1.0], bins, alpha=0.75, label="BP = 1")
        axtrain_roc.hist(valid_pred[label_valid == 0.0], bins, alpha=0.75, label="SZ = 0")
        axtrain_roc.legend()
        axvalid_roc.set_title("ROC on valid")
        axvalid_roc.plot(fpr, tpr, label="AUC = "+str(AUC)[:6])
        axvalid_roc.legend()

    axtest_dist.set_title("Probability distribution on test")
    axtest_dist.hist(test_pred, bins, label="SZ+BP")
    axtest_dist.legend()
    plt.show()