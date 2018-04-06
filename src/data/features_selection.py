# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 19:02:33 2018

@author: sandr
"""

import make_dataset
import build_data
import pandas as pd
import numpy as np

import sklearn.feature_selection as fs
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from time import time
import math


# import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
import sklearn.model_selection as ms
from sklearn import metrics
from sklearn.utils.extmath import density
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn import svm


RANDOM_SEED = 123


def get_attributes(data, support):
    columns = list(data.columns.values)
    attrib = [columns[i] for i in list(support.nonzero()[0])]
    return attrib

def select_specific_features (all_data, columns):    
    
    red_data = all_data.filter(columns, axis = 1)
    types = red_data.columns.to_series().groupby(red_data.dtypes).groups
    ctext = types[np.dtype('object')]
    for i in ctext:
        red_data[i], _ = pd.factorize(red_data[i], sort=True)
    
    labels = red_data['MBA_DELINQUENCY_STATUS_next']
    red_data.drop(['MBA_DELINQUENCY_STATUS_next'], axis=1, inplace=True) 
    return red_data, labels

    
def attributes_selection(red_data, labels, k=50, invariant=True, function=f_classif):
    
    def inv_select(data, labels, k, function):        
        # selected = SelectKBest(function, k=k).fit(data, labels)   #fs.SelectFwe(function, alpha=0.05) , fs.SelectPercentile(function, percentile=10) % features to keep     
        # selected = fs.SelectFwe(function, alpha=0.05).fit(data, labels)
        selected = fs.SelectPercentile(function, percentile=80).fit(data, labels)
        return selected.pvalues_, selected.get_support()
    
    def rec_select(data, labels, k):        
        classifier = ExtraTreesClassifier() # by random forest by default n=10
        model = RFE(classifier, k) # recursive trainner
        selected = model.fit(data, labels)        
        return selected.ranking_, selected.support_
    
       
    if invariant:
        # by Univariant SelectKBest:
        support = inv_select(red_data, labels, k, function)
    else:
        support = rec_select(red_data,labels,k)
        
    columns = list(red_data.columns.values)
    attrib = [columns[i] for i in list(support[1].nonzero()[0])]
    return attrib, support[0]

def normalize_coef(selector, isEstimator=False):
    if isEstimator:
        scores = (selector.coef_ ** 2).sum(axis=0)   #only available in the case of a linear kernel.      
    else:
        scores = -np.log10(selector.pvalues_)            
    scores /= scores.max()
    return scores

#it doesnt work:
def svm_classify(x_train, y_train):
    clf = svm.SVC(verbose=True)
    y_train = y_train.as_matrix()  # pd.reshape(y_train.as_matrix(), (y_train.shape[0], 1))
    clf.fit(x_train.as_matrix(), y_train)
    return clf

def print_dec_matrix(conf_mtx):
    for conf_row in conf_mtx:
        for conf_value in conf_row:
            print('{:.6f}'.format(conf_value), '\t', end='')
        print('|\n', end='')        
    print('--------------------------------------------------------------'
          '------------')

def confusion_matrix(real_labels, preds_label, message):
    print(message)
    cm = metrics.confusion_matrix(real_labels, preds_label)
    cm = cm.astype('float') / cm.sum(axis=1)
    # cm = cm.round(decimals=6)
    print_dec_matrix(cm)
    print('Better ACC={:.4f}'.format(cm.diagonal().mean()))
    
# Benchmark classifiers
def benchmark(clf, name,X_train, y_train, X_test, y_test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    
    t0 = time()
    training_pred = clf.predict(X_train)
    train_time = time() - t0
    print("training Prediction time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test prediction time:  %0.3fs" % test_time)

    score1 = clf.score(X_train, y_train)
    print("ACC training set:   %0.3f" % score1)
    score = metrics.accuracy_score(y_test, pred)
    print("ACC testing set:   %0.3f" % score)            
    
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_pred, y_train))
    print("Root Mean Squared Error in Training Set:   %0.3f" % training_root_mean_squared_error)
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(pred, y_test))
    print("Root Mean Squared Error in Test Set:   %0.3f" % validation_root_mean_squared_error)

    if hasattr(clf.named_steps[name], 'feature_importances_'):
        print("dimensionality: %d" % clf.named_steps[name].feature_importances_.shape)
        print("density: %f" % density(clf.named_steps[name].feature_importances_))
        
    if hasattr(clf.named_steps[name], 'coef_'):
        print("dimensionality: %d" % clf.named_steps[name].coef_.shape[1])
        print("density: %f" % density(clf.named_steps[name].coef_))
                
    print("classification report testing set:")
    print(metrics.classification_report(y_test, pred)) # , target_names=target_names))
    
    
    if (y_test.shape[1] > 1): # multilabel-indicator is not supported        
        y_train_labels = np.array(build_data.encode_binary_to_labeled_column(pd.DataFrame(y_train)))
        pred_train_labels = np.array(build_data.encode_binary_to_labeled_column(pd.DataFrame(training_pred)))
        confusion_matrix(y_train_labels, pred_train_labels, "confusion matrix training set:")
                
        y_test_labels = np.array(build_data.encode_binary_to_labeled_column(pd.DataFrame(y_test)))
        pred_labels = np.array(build_data.encode_binary_to_labeled_column(pd.DataFrame(pred)))        
        confusion_matrix(y_test_labels, pred_labels, "confusion matrix testing set:")
    else:
        confusion_matrix(y_train, training_pred, "confusion matrix training set:")
        confusion_matrix(y_test, pred, "confusion matrix testing set:")    
    
    clf_descr = str(clf).split('(')[0]
    return clf_descr, training_pred, pred, score, train_time, test_time

def apply_classifier(X_train, X_test, y_train, y_test, random_seed):    
    clf = Pipeline([('RandomForestClassifier', RandomForestClassifier(n_estimators=100, n_jobs = -1, random_state=random_seed))])
    results = benchmark(clf, 'RandomForestClassifier', X_train, y_train, X_test, y_test)
    return results

def selection_process(b_data, labels):   
        
    # scatter_variables(all_data, 'MONTH')
    
    # n_data = normalize(b_data)
    # The idea is to compare possible outcomes combination.    
    
    # this works:
    # X_train, X_test, y_train, y_test = ms.train_test_split(b_data, labels, test_size=0.33, random_state=RANDOM_SEED, stratify=labels)    
    # clf = Pipeline([('RandomForestClassifier', RandomForestClassifier(n_estimators=100, n_jobs = -1, random_state=RANDOM_SEED))])
    # results = benchmark(clf, 'RandomForestClassifier', X_train, y_train, X_test, y_test)
    
    
    # for i in all_data.columns.values:
    #    print(all_data[i].min(), '-', all_data[i].max())
    # clf = svm_classify(X_train, y_train)
    # y_pred = clf.predict(X_test)    
    # print(classification_report(y_test, y_pred))
    
    # this doesnt work:    
    # attrib = attributes_selection(b_data, labels, k=50, invariant=True, function=f_classif) # mutual_info_classif
    # red_data = b_data[attrib[0]]    
    # n_data = normalize(red_data)
    # X_train, X_test, y_train, y_test = ms.train_test_split(n_data, labels, test_size=0.33, random_state=RANDOM_SEED)    
    #results = benchmark(clf, X_train, y_train, X_test, y_test)
    # sel_clf = svm_classify(X_train, y_train)
    # sel_y_pred = sel_clf.predict(X_test)
    # print(classification_report(y_test, sel_y_pred))
    
    
    for  name, slt in (('LinearSVCselection', SelectFromModel(LinearSVC(penalty="l1"))),
                       ('SelectKBest', SelectKBest(mutual_info_classif, k=50)),
                       ('SelectFwe', fs.SelectFwe(mutual_info_classif, alpha=0.05)),
        ):
        print('&' * 80)
        print(name)
        pip = Pipeline([slt])        
        results = []
        for name, clf in (
            ("RidgeClassifier", RidgeClassifier(tol=1e-2, solver="lsqr")),
            ("Perceptron", Perceptron(n_iter=50)),
            ("PassiveAggressive", PassiveAggressiveClassifier(n_iter=50)),
            ("kNN", KNeighborsClassifier(n_neighbors=10)),
            ("RandomForest", RandomForestClassifier(n_estimators=100))): 
            print('=' * 80)
            print(name)
            pip.classes_.append(clf)
            results.append(benchmark(pip, name, X_train, y_train, X_test, y_test))            


columns = ['MBA_DAYS_DELINQUENT','CURRENT_INTEREST_RATE', 'LLMA2_CURRENT_INTEREST_SPREAD', 'LOANAGE', 
                          'CURRENT_BALANCE', 'SCHEDULED_PRINCIPAL', 'SCHEDULED_MONTHLY_PANDI','LLMA2_HIST_LAST_12_MONTHS_MIS', 
                          'LLMA2_C_IN_LAST_12_MONTHS','LLMA2_30_IN_LAST_12_MONTHS', 'LLMA2_60_IN_LAST_12_MONTHS',
                          'LLMA2_90_IN_LAST_12_MONTHS', 'LLMA2_FC_IN_LAST_12_MONTHS','LLMA2_REO_IN_LAST_12_MONTHS', 
                          'FICO_SCORE_ORIGINATION', 'INITIAL_INTEREST_RATE', 'ORIGINAL_LTV','ORIGINAL_BALANCE', 
                          'BACKEND_RATIO', 'SALE_PRICE','NUMBER_OF_UNITS', 'LLMA2_APPVAL_LT_SALEPRICE','LLMA2_ORIG_RATE_SPREAD', 
                          'MARGIN', 'PERIODIC_RATE_CAP','PERIODIC_RATE_FLOOR', 'LIFETIME_RATE_CAP', 'LIFETIME_RATE_FLOOR',
                          'RATE_RESET_FREQUENCY', 'PAY_RESET_FREQUENCY','FIRST_RATE_RESET_PERIOD', 'LLMA2_PRIME', 'LLMA2_SUBPRIME', 
                          'OCCUPANCY_TYPE_1', 'OCCUPANCY_TYPE_2','OCCUPANCY_TYPE_3', 'OCCUPANCY_TYPE_U', 'PRODUCT_TYPE_10',
                          'PRODUCT_TYPE_20', 'PRODUCT_TYPE_30', 'PRODUCT_TYPE_50','PRODUCT_TYPE_51', 'PRODUCT_TYPE_52', 
                          'PRODUCT_TYPE_54','PRODUCT_TYPE_5A', 'PRODUCT_TYPE_5Z', 'PRODUCT_TYPE_60','PRODUCT_TYPE_63', 
                          'PRODUCT_TYPE_70', 'PRODUCT_TYPE_80','PRODUCT_TYPE_81', 'PRODUCT_TYPE_82', 'PRODUCT_TYPE_83','PRODUCT_TYPE_84', 
                          'PRODUCT_TYPE_8Z', 'PRODUCT_TYPE_U','LOAN_PURPOSE_1', 'LOAN_PURPOSE_2', 'LOAN_PURPOSE_3','LOAN_PURPOSE_5', 
                          'LOAN_PURPOSE_6', 'LOAN_PURPOSE_7','LOAN_PURPOSE_8', 'LOAN_PURPOSE_9', 'LOAN_PURPOSE_B','LOAN_PURPOSE_U', 
                          'DOCUMENTATION_TYPE_1', 'DOCUMENTATION_TYPE_2','DOCUMENTATION_TYPE_3', 'DOCUMENTATION_TYPE_U', 'CHANNEL_1', 
                          'CHANNEL_2', 'CHANNEL_3', 'CHANNEL_4', 'CHANNEL_7', 'CHANNEL_8','CHANNEL_9', 'CHANNEL_A', 'CHANNEL_C', 'CHANNEL_D', 
                          'CHANNEL_U','LOAN_TYPE_1', 'LOAN_TYPE_2', 'LOAN_TYPE_3', 'LOAN_TYPE_4','LOAN_TYPE_U', 'STATE', 'UR', 'MBA_DELINQUENCY_STATUS_next']
# b_data = select_specific_features(all_data, columns)    
# X_train, X_test, y_train, y_test = ms.train_test_split(b_data, labels, test_size=0.33, random_state=RANDOM_SEED, stratify=labels)    

#def get_data(training_examples, validation_examples, test_examples, weighted_sampling, 
#             dataset_name='MORT', stratified_flag=False, refNorm=True):
#    return make_dataset.get_data(training_examples, validation_examples, test_examples, weighted_sampling, 
#                                 dataset_name=dataset_name, stratified_flag = stratified_flag, refNorm=refNorm)