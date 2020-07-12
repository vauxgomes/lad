#!/usr/bin/env python

'''
Template
	https://github.com/scikit-learn-contrib
'''

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report

from binarizer.cutpointbinarizer import CutpointBinarizer
from featureselection.featureselection import GreedySetCover
from rulegenerator.maxpatterns import MaxPatterns

# Docs
__author__ = 'Vaux Gomes'
__version__ = '0.1.0'


class LADClassifier(BaseEstimator):

    def __init__(self, tolerance=0.0, purity=0.95):
        '''
            LAD Classifier Constructor

            Implements the Maximized Prime Patterns heuristic described in the
            "Maximum Patterns in Datasets" paper. It generates one pattern (rule)
            per observation, while attempting to: (i) maximize the coverage of other
            observations belonging to the same class, and (ii) preventing the
            coverage of too many observations from outside that class. The amount of
            "outside" coverage allowed is controlled by the minimum purity parameter
            (from the main LAD classifier).

            Parameter:
            tolerance -- Cutpoint tolerance. It must be bigger than or equal to zero
            purity -- Minimum rule purity
        '''

        self.tolerance = tolerance
        self.purity = purity
        
        self.__rules = None
        self.__cutpoints = None
        self.__selected = None
        self.__labels = None

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        
        weights = {}

        for r in self.__rules:

            label = r['label']
            weight = r['weight']

            indexes = np.arange(X.shape[0])

            for i, condition in enumerate(r['conditions']):
                att = r['attributes'][i]
                val = r['values'][i]

                if (condition):
                    #print(f'att{att} <= {val}', end=', ')
                    indexes = indexes[np.where(X.T[att, indexes] <= val)]
                else:
                    #print(f'att{att} > {val}', end=', ')
                    indexes = indexes[np.where(X.T[att, indexes] > val)]

            #print(r['label'])

            for i in indexes:
                weights[i] = weights.get(i, {})
                weights[i][label] = weights[i].get(label, 0) + weight

        pred = []
        for i in range(X.shape[0]):
            if i not in weights:
                pred.append(2)
            else:
                pred.append(max(weights[i], key=weights[i].get))
                
        return np.array(pred)

    def fit(self, X, y):        
        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True

        #print('# Binarization')
        cpb = CutpointBinarizer(self.tolerance)
        Xbin = cpb.fit_transform(X, y)

        #print('# Feature Selection')
        gsc = GreedySetCover()
        Xbin = gsc.fit_transform(Xbin, y)

        #print('# Rule building')
        mxp = MaxPatterns(self.purity)
        mxp.fit(Xbin, y)
        
        #
        self.__labels = np.unique(y)
        self.__cutpoints = cpb.get_cutpoints()
        self.__selected = gsc.get_selected()
        self.__rules = mxp.get_rules()
        
        #print('# Convert binary rules into numeric rules')
        self.__rules_to_numerical()

        return self  # `fit` should always return `self`
    
    def __rules_to_numerical(self):
        BIGGER_THAN = '>'
        LESS_EQUAL_THAN = '<='

        for r in self.__rules:
            cutpoints = [self.__cutpoints[i] for i in self.__selected[r['attributes']]]
            
            r['attributes'].clear()
            r['values'] = []
            
            for i, c in enumerate(cutpoints):
                r['attributes'].append(c[0])
                r['values'].append(c[1])

'''
if __name__ == '__main__':
    # Load
    df = pd.read_csv('data/iris.data', names='att0 att1 att2 att3 class'.split())
    df = df.sample(frac=1, random_state=0) # Shuffle

    # Sampling
    sample_size = int(0.9*len(df))

    # Train
    X = df.iloc[:sample_size, :-1]
    y = df.iloc[:sample_size, -1]

    print(X.loc[[114,  62,  33, 107,   7, 100,  40]])

    # Test
    X_test = df.iloc[sample_size + 1:, :-1]
    y_test = df.iloc[sample_size + 1:, -1]

    # Classifier
    lad = LADClassifier()
    lad.fit(X, y)

    # Prediction
    y_hat = lad.predict(X_test)

    # Report
    print(classification_report(y_test, y_hat))
'''