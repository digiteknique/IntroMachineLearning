#!/usr/bin/python
import sys
import math
sys.path.append("./tools/")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

# Method to run select k best features 
def select_kbest(features, labels):
    #Stratified split
    split = StratifiedKFold().split(features, labels)
    best_score = 0.0
    best_params = None
    best_features = None
    feature_scores = None
    for k, (train, test) in enumerate(split):
        features_train, features_test, labels_train, labels_test = features[train], features[test], labels[train], labels[test]

        param_grid = {
            'kbest__k': [4, 6, 8, 10, 12, 14],
        }
        kbest = SelectKBest(f_classif)
        clf = AdaBoostClassifier(algorithm='SAMME', n_estimators=5)
        pipeline = Pipeline([
                              ('kbest', kbest), 
                              ('clf', clf)
                            ])

        cv_kmeans = GridSearchCV(pipeline, param_grid=param_grid, scoring='f1')
        cv_kmeans.fit(features_train, labels_train)

        param = cv_kmeans.best_params_
        score =  cv_kmeans.best_score_

        if score > best_score:
            best_score = score
            best_params = param
            best_features = cv_kmeans.best_estimator_.named_steps['kbest'].get_support()
            feature_scores = cv_kmeans.best_estimator_.named_steps['kbest'].scores_
    print best_score
    print best_params
    print best_features
    print feature_scores


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
                  

def create_features(data_dict):
    for entry in data_dict:
        poi_messages_to = data_dict[entry]['from_this_person_to_poi']  
        poi_messages_from = data_dict[entry]['from_poi_to_this_person']  
        poi_messages_shared = data_dict[entry]['shared_receipt_with_poi']
        messages_from = data_dict[entry]['from_messages']
        messages_to = data_dict[entry]['to_messages']
        data_dict[entry]['email_ratio'] = message_ratio((poi_messages_from + poi_messages_shared + poi_messages_to), (messages_from + messages_to))

        restricted = data_dict[entry]['restricted_stock']
        restricted_deferred = data_dict[entry]['restricted_stock_deferred']
        total_stock = data_dict[entry]['total_stock_value']
        data_dict[entry]['stock_ratio'] =stock_ratio(restricted, restricted_deferred, total_stock)

    return data_dict