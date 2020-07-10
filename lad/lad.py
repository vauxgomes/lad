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
        
        self.__rules = []
        self.__cutpoints = []
        self.__labels = None

    def predict(self, X):
        _ = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        
        # Auxiliary columns
        for label in self.__labels:
            X[label] = 0.0

        # Rules coverage
        for r in self.__rules[:]:
            query = ' & '.join([f'{att}{condition}{val}' for att, condition, val in r['conditions']])
            indexes = X.query(query).index

            weight = r['weight']
            label = r['label']

            X.loc[indexes, label] += weight

        # return np.ones(X.shape[0], dtype=np.int64)
        return X[self.__labels].eq(X[self.__labels].max(1), axis=0).dot(X[self.__labels].columns)

    def fit(self, X, y):        
        # Template stuff
        # -------------------------------
        _, _ = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True
        # -------------------------------

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
        self.__labels = list(y.unique())
        self.__cutpoints = cpb.get_cutpoints()
        self.__rules = mxp.get_rules()
        
        #print('# Convert binary rules into numeric rules')
        self.__rules_to_numerical()

        return self  # `fit` should always return `self`
    
    def __rules_to_numerical(self):
        BIGGER_THAN = '>'
        LESS_EQUAL_THAN = '<='

        for r in self.__rules:        
            r['conditions'] = []

            for att, val in zip(r['attributes'], r['values']):
                condition = LESS_EQUAL_THAN if val else BIGGER_THAN
                att, val = self.__cutpoints[att] # Convertion
                r['conditions'].append((att, condition, val))

'''
if __name__ == '__main__':
    # Load
    df = pd.read_csv('data/iris.data', names='att0 att1 att2 att3 class'.split())
    df = df.sample(frac=1, random_state=0) # Shuffle

    # Sampling
    sample_size = int(0.5*len(df))

    # Train
    X = df.iloc[:sample_size, :-1]
    y = df.iloc[:sample_size, -1]

    # Test
    X_test = df.iloc[sample_size + 1:, :-1]
    y_test = df.iloc[sample_size + 1:, -1]

    # Classifier
    lad = LADClassifier()
    lad.fit(X, y)

    # Prediction
    y_hat = lad.predict(X_test)

    # Report
    print(classification_report(y_test, y_hat, target_names=list(y.unique())))
'''