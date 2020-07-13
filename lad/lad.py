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
from rulegenerator.maxpatterns import MaxPatterns, LazyMaxPatterns

# Docs
__author__ = 'Vaux Gomes'
__version__ = '0.1.0'


class LADClassifier(BaseEstimator):

    def __init__(self, tolerance=0.0, purity=0.95, mode="eager"):
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
        self.mode = mode

        self.rule_model = None

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True

        # print('# Binarization')
        cpb = CutpointBinarizer(self.tolerance)
        Xbin = cpb.fit_transform(X, y)

        # print('# Feature Selection')
        gsc = GreedySetCover()
        Xbin = gsc.fit_transform(Xbin, y)

        # print('# Rule building')
        if self.mode == 'eager':
            self.rule_model = MaxPatterns(self.purity)

        elif self.mode == 'lazy':
            self.rule_model = LazyMaxPatterns(self.purity)
        
        self.rule_model.fit(Xbin, y)
        self.rule_model.adjust(cpb, gsc)

        return self  # `fit` should always return `self`

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        return self.rule_model.predict(X)

'''
if __name__ == '__main__':
    # Load
    #df = pd.read_csv('data/iris.data', names='att0 att1 att2 att3 class'.split())
    #df = df.sample(frac=1, random_state=0)  # Shuffle

    df = pd.read_csv('data/bcw.data', names='id att0 att1 att2 att3 att4 att5 att6 att7 att8 class'.split())
    df['att5'] = df['att6'].replace('?', np.NaN).astype(int) # Trocando ? por NaN
    df = df.drop(columns=['id'])
    df = df.dropna() # Eliminando linhas que contém NaN
    df = df.sample(frac=1, random_state=0) # Shuffle

    # Sampling
    sample_size = int(0.9*len(df))

    # Train
    X = df.iloc[:sample_size, :-1]
    y = df.iloc[:sample_size, -1]

    # Test
    X_test = df.iloc[sample_size + 1:, :-1]
    y_test = df.iloc[sample_size + 1:, -1]

    # Classifier
    lad = LADClassifier(mode='lazy')
    lad.fit(X, y)

    # Prediction
    y_hat = lad.predict(X_test)

    print(y_hat)

    # Report
    print(classification_report(y_test, y_hat))
'''