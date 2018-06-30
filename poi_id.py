#!/usr/bin/python
import sys
sys.path.append("./tools/")

import pickle
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from feature_selection import select_kbest, tree_selection, scale_features, message_ratio, stock_ratio
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# All features except email_address
all_features_list = ['poi',  'salary', 'to_messages', 'deferral_payments',
                 'total_payments', 'loan_advances', 'bonus', 
                 'restricted_stock_deferred', 'deferred_income',
                 'total_stock_value', 'expenses', 'from_poi_to_this_person',
                 'exercised_stock_options',  'from_messages',            
                 'other', 'from_this_person_to_poi', 'long_term_incentive',      
                 'shared_receipt_with_poi', 'restricted_stock',
                 'director_fees' ]

intuition_feature_list = ['poi', 'salary', 'bonus', 
                          'expenses', 'deferral_payments' ]

kbest_feature_list = ['poi', 'salary', 'total_payments', 'bonus', 'total_stock_value', 
                      'exercised_stock_options']

tree_feature_list = ['poi','salary', 'total_payments', 'bonus', 'deferred_income',
                     'expenses' ]

features_list = tree_feature_list


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)



### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
my_dataset.pop('TOTAL', 0) # Remove the total line generated from the spreadsheet
my_dataset.pop('LOCKHART EUGENE E', 0) # Remove LOCKHART EUGENE E for all 0s
my_dataset.pop('THE TRAVEL AGENCY IN THE PARK', 0) # Remove THE TRAVEL AGENCY IN THE PARK for not being relavent

#Create features
# variance_threshold(my_dataset)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# kmeans
# data = scale_features(data)
# labels, features = targetFeatureSplit(data)
# clf = KMeans(n_clusters=2)

# RandomForest
clf = RandomForestClassifier(random_state=42)

#AdaBoost
# clf = AdaBoostClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Small method to tune parameters using GridSearchCV
def tune_classifier(clf, param_grid, features_train, labels_train):
    # Use GridSearchCV tune, this takes a while
    cv_rfc = GridSearchCV(estimator=clf, param_grid=param_grid)
    cv_rfc.fit(features_train, labels_train)

    print cv_rfc.best_params_
    return cv_rfc.best_estimator_

def tune_ada_classifier(features_train, labels_train):
    param_grid = {
        'algorithm': ['SAMME', 'SAMME.R'],    
        'n_estimators': [5, 10, 50, 200, 500, 1000, 2000]
    }
    return tune_classifier(AdaBoostClassifier(), param_grid, features_train, labels_train)

def tune_rf_classifier(features_train, labels_train):
    param_grid = {
        'n_estimators' : [ 5, 10, 50, 200, 500, 1000, 2000 ],
        # 'max_features': [ 'auto', 'sqrt', 'log2' ],
        # 'max_depth': [ 5, 6, 7, 8 ],
        # 'criterion': [ 'gini', 'entropy' ],
        'min_samples_split': [ 2, 4, 5, 10 ]
    }
    return tune_classifier(RandomForestClassifier(), param_grid, features_train, labels_train)

# clf = tune_rf_classifier(features_train, labels_train)
clf = tune_ada_classifier(features_train, labels_train)

# Optimized RF with intuition features Acc: 0.78850 Prec: 0.30758 Rec: 0.21500
# clf = RandomForestClassifier(min_samples_split=2, n_estimators=5)

# Optimized RF with kbest features Acc: 0.86773 Prec: 0.50866 Rec: 0.23500
# clf = RandomForestClassifier(min_samples_split=2, n_estimators=10)

# Optimized RF with tree features Acc: 0.83585 Prec: 0.40822 Rec: 0.14900
# clf = RandomForestClassifier(min_samples_split=10, n_estimators=10)

# Optimized adaboost with intuition features Acc: 0.84617 Prec: 0.54578 Rec: 0.45900
# clf = AdaBoostClassifier(n_estimators=5, algorithm='SAMME')

# Optimized adaboost with kbest features Acc: 0.86760 Prec: 0.50928 Rec: 0.19200 
# clf = AdaBoostClassifier(n_estimators=5, algorithm='SAMME.R')

# Optimized adaboost with tree features Acc: 0.83008 Prec: 0.42029 Rec: 0.27550 
# clf = AdaBoostClassifier(n_estimators=500, algorithm='SAMME')


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

