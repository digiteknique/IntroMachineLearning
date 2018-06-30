#!/usr/bin/python
import sys
import math
sys.path.append("./tools/")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

# Method to run select k best features 
def select_kbest(features_train, labels_train):
    best = SelectKBest(f_classif, 6).fit(features_train, labels_train)
    print best.scores_
    print best.get_support()

# Method to run tree selection
def tree_selection(features_train, labels_train): 
    clf = ExtraTreesClassifier(n_estimators=250, random_state=0)
    clf = clf.fit(features_train, labels_train)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print indices
    
    print clf.feature_importances_
    model = SelectFromModel(clf, prefit=True)
    print model.get_support()

# Method to scale features
def scale_features(arr):  
    for i in range(1,arr.shape[1]):
        arrmin = min(arr[:,i])
        arrmax = max(arr[:,i])
        if arrmin == arrmax:
            arr[:,i] = arr[:,i]/arrmin
        else:
            arr[:,i] = (arr[:,i]-arrmin)/(arrmin-arrmax)
    return arr  
       
      
# Method to create ratio of Messages involving a POI and all messages
def message_ratio(poi, all_messages):
    # If either value have a NaN we can't compute the ratio, same if all is 0
    if (isinstance(poi, (int, long)) and \
        isinstance(all_messages, (int, long))) == False:
        return 0
    if all_messages == 0:
        return 0
    
    return 1.0 * poi / all_messages

def stock_ratio(restricted, restricted_deferred, total):
    if (isinstance(restricted, (int, long)) and \
        isinstance(restricted_deferred, (int, long)) and \
        isinstance(total, (int, long))) == False:
        return 0
    
    if total == 0:
        return 0

    return 1.0 * (restricted + restricted_deferred) / total
                  
           