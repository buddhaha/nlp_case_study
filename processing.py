#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Foobar.py: Description of what foobar does."""

__author__ = "Mirek Buddha"

import os
import sys
import argparse
import requests
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
import progressbar # optional
from time import sleep
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from ast import literal_eval
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import svm
from time import time

SGDC_GRID = {'fit_intercept': [True, False],
             'early_stopping': [True, False],
             'loss': ['hinge', 'log', 'squared_hinge'],
             'penalty': ['l2', 'l1', 'none']
             }
MNB_GRID = {'alpha': [0.01, 0.1, 0.3, 0.5, 0.75, 0.8, 0.85, 0.9, 1],
            'fit_prior' : [True,False]
            }

SVM_GRID = {'C': [1, 10, 100],  # 0.1, 1000
#        'class_prior': [True, False],
       'degree' : [2,3], ### ,4
       'kernel' : ['linear'], #, 'poly', 'rbf', 'sigmoid'], # , 'precomputed' removed due to "ValueError: X should be a square kernel matrix"
       'gamma' : ['scale', 'auto'],
       ##'shrinking' : [True,False],
       'probability' : [True], # False
       'verbose' : [True]  # ,False
       #,'decision_function_shape' : ['ovo', 'ovr'] ### ignored for binary
       ##,'break_ties' : [True,False]
        }

RFC_GRID = {
    'n_estimators': [10, 200, 500],
    'max_features': ['auto'],  ## , 'sqrt', 'log2'
    'max_depth' : [4,6,8],
    'criterion' :['entropy'] # 'gini',
    }


def load_auth_key():
    with open('deepl_authkey.txt') as f:
        key = f.readline()
    return key

_PERSONAL_AUTH_KEY = load_auth_key()

def translate_w_deepl(text):
    data = {
        'auth_key': _PERSONAL_AUTH_KEY,
        'text': text,
        'target_lang': 'EN'
    }
    response = requests.post('https://api-free.deepl.com/v2/translate', data=data)
    return response.json()['translations'][0]['text']

DEEPL_KEY = load_auth_key()
RAW_DATA_PATH = r"raw_data/Relevant vs Irrelevant.xlsx"

def translate_to_eng(df):
    df['eng_title'] = str()
    df['eng_desc'] = str()
    df['eng_text'] = str()

    for i, r in df.head(20).iterrows():
        bar.update(i)
        sleep(0.1)
        # print(r.title)
        # print(r.description)
        # print(r.maintext)
        # print(df.loc[i, "maintext"])
        # '''
        if r.language == 'en':
            df.loc[i, 'eng_title'] = r.title
            df.loc[i, 'eng_desc'] = r.description
            df.loc[i, 'eng_text'] = r.maintext
        else:
            # print('title', r.title)#,'maintext']])
            # print('description', r.description)  # ,'maintext']])
            df.loc[i, 'eng_title'] = translate_w_deepl(r.title)
            df.loc[i, 'eng_desc'] = translate_w_deepl(r.description)
            df.loc[i, 'eng_text'] = translate_w_deepl(r.maintext)
        # '''
    df.to_csv(r"raw_data/eng_version.csv")
    return df


def df_preprocessing_lemma_cols(df):
    print('..... preprocessing data .....')
    print('..... initial stats .....')
    print('initial shape:', df.shape)
    print('columns:', df.columns)
    print('target value counts:', df.target.value_counts())
    count = 0
    idx_to_drop = []
    for index, row in df.iterrows():
        if not row['lemma_maintext'].endswith(']'):
            #print('index:', index, 'row', row['lemma_maintext'][:-3])
            last_row = row['lemma_maintext']
            idx_to_drop.append(index)
            count += 1
        #print('index:', index, 'row', eval(row['lemma_maintext']))
    #print(len(df.loc[23,'lemma_maintext']))
    #print(eval(df.loc[23,'lemma_maintext']))
    print('wrongly parsed data:',count)
    ## only 19 rows, just drop them
    df.drop(idx_to_drop, inplace=True)
    #print(eval(last_row).split(',')[:-2])
    #print(type(eval(df.lemma_maintext.loc[999])))
    count = 0
    idx_to_drop = []
    for index, row in df.iterrows():
        if not row['lemma_maintext'].endswith(']'):
            #print('index:', index, 'row', row['lemma_maintext'][:-3])
            last_row = row['lemma_maintext']
            idx_to_drop.append(index)
            count += 1
    print('wrongly parsed data:',count)
    df.lemma_title = df.lemma_title.apply(eval)
    df.lemma_description = df.lemma_description.apply(eval)
    df.lemma_maintext = df.lemma_maintext.apply(eval)
    #sample['lemma_title'] = sample['lemma_title'].apply(literal_eval)
    df.lemma_title = [','.join(map(str, l)) for l in df.lemma_title]
    df.lemma_description = [','.join(map(str, l)) for l in df.lemma_description]
    df.lemma_maintext = [','.join(map(str, l)) for l in df.lemma_maintext]


    df = df[df.language == 'en'][['lemma_title', 'lemma_description', 'lemma_maintext', 'target']]
    #df = df[]
    print('..... after easy processing data .....')
    print('..... final stats .....')
    print('final shape:', df.shape)
    print('columns:', df.columns)
    print('target value counts:', df.target.value_counts())
    return df

def extract_features(df):
    # define column transformer
    # and feature extraction

    return df

def main():

    df = pd.read_excel(RAW_DATA_PATH)
    df = df_preprocessing_lemma_cols(df)
    feature_cols = ['lemma_title', 'lemma_description', 'lemma_maintext']
    features = df[feature_cols].copy()
    X_train, X_test, y_train, y_test = train_test_split(features, df['target'], test_size=0.4, random_state=666)
    print("train: \n", y_train.value_counts())
    print("test: \n", y_test.value_counts())
    MAX_TFIDF_FEATURES = 5000
    tfidf_vect = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES, ngram_range=(1, 3))
    preprocess = ColumnTransformer(
        [('lemma_title_tfidf', tfidf_vect, 'lemma_title'),
         ('lemma_description_tfidf' , tfidf_vect, 'lemma_description'),
         ('lemma_maintext_tfidf', tfidf_vect, 'lemma_maintext')
         ],
          remainder='drop', verbose_feature_names_out=True)

    # preprocess.fit(df)
    # STOCHASTIC GRADIENT DESCENT
    t = time()
    model_sgdc = make_pipeline(
        preprocess,
        GridSearchCV(SGDClassifier(random_state=666), param_grid=SGDC_GRID, cv=5, refit=True) ### set: refit=True!!!
    )
    model_sgdc.fit(X_train, y_train)
    training_time = time() - t
    model_sgdc_y_pred = model_sgdc.predict(X_test)

    print(metrics.classification_report(y_test, model_sgdc_y_pred, target_names=['Positive', 'Negative']))
    print('#########################------------------------------')
    print('training time: {:.1f} s'.format(training_time))
    print('------------------------------')
    print('model_sgdc.best_params_', model_sgdc[1].best_params_)
    print('------------------------------')
    print('accuracy', metrics.accuracy_score(y_test, model_sgdc_y_pred))
    print('------------------------------')
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, model_sgdc_y_pred))
    print('#########################------------------------------')
    ### this performance seems to be quite good.

    # multinomial Naive Bayes
    t = time()
    model_mnb = make_pipeline(
        preprocess,
        GridSearchCV(MultinomialNB(), param_grid=MNB_GRID, cv=5, refit=True) ### set: refit=True!!!
    )
    model_mnb.fit(X_train, y_train)
    training_time = time() - t
    model_mnb_y_pred = model_sgdc.predict(X_test)

    print(metrics.classification_report(y_test, model_mnb_y_pred, target_names=['Positive', 'Negative']))
    print('#########################------------------------------')
    print('training time: {:.1f} s'.format(training_time))
    print('------------------------------')
    print('model_sgdc.best_params_', model_mnb[1].best_params_)
    print('------------------------------')
    print('accuracy', metrics.accuracy_score(y_test, model_mnb_y_pred))
    print('------------------------------')
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, model_mnb_y_pred))
    print('#########################------------------------------')

    # svm
    t = time()
    model_svm = make_pipeline(
        preprocess,
        GridSearchCV(svm.SVC(), param_grid=SVM_GRID, cv=5, refit=True) ### set: refit=True!!!
    )
    model_svm.fit(X_train, y_train)
    training_time = time() - t
    model_svm_y_pred = model_svm.predict(X_test)

    print(metrics.classification_report(y_test, model_svm_y_pred, target_names=['Positive', 'Negative']))
    print('#########################------------------------------')
    print('training time: {:.1f} s'.format(training_time))
    print('------------------------------')
    print('model_sgdc.best_params_', model_svm[1].best_params_)
    print('------------------------------')
    print('accuracy', metrics.accuracy_score(y_test, model_svm_y_pred))
    print('------------------------------')
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, model_svm_y_pred))
    print('#########################------------------------------')



    # RandomForestClassifier
    t = time()
    model_rfc = make_pipeline(
        preprocess,
        GridSearchCV(RandomForestClassifier(random_state = 666), param_grid=RFC_GRID, cv=5, refit=True) ### set: refit=True!!!
    )
    model_rfc.fit(X_train, y_train)
    training_time = time() - t
    model_rfc_y_pred = model_rfc.predict(X_test)

    print(metrics.classification_report(y_test, model_rfc_y_pred, target_names=['Positive', 'Negative']))
    print('#########################------------------------------')
    print('training time: {:.1f} s'.format(training_time))
    print('------------------------------')
    print('model_sgdc.best_params_', model_rfc[1].best_params_)
    print('------------------------------')
    print('accuracy', metrics.accuracy_score(y_test, model_rfc_y_pred))
    print('------------------------------')
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, model_rfc_y_pred))
    print('#########################------------------------------')

if __name__ == '__main__':
    main()
