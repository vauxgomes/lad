#!/usr/bin/env python

'''
References:
https://scikit-learn.org/stable/developers/develop.html
https://sklearn-template.readthedocs.io/en/latest/quick_start.html
'''


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from lad.binarizer.cutpointbinarizer import CutpointBinarizer
from lad.featureselection.featureselection import GreedySetCover
from lad.rulegenerator.eager import MaxPatterns
from lad.rulegenerator.lazy import LazyPatterns

# Docs
__author__ = 'Vaux Gomes'
__version__ = '0.4'


class LADClassifier(BaseEstimator, ClassifierMixin):
    '''
        LAD Classifier

        Implements the Maximized Prime Patterns heuristic described in the
        "Maximum Patterns in Datasets" paper. It generates one pattern (rule)
        per observation, while attempting to: (i) maximize the coverage of other
        observations belonging to the same class, and (ii) preventing the
        coverage of too many observations from outside that class. The amount of
        "outside" coverage allowed is controlled by the minimum purity parameter
        (from the main LAD classifier).

        Attributes
        ---------
        tolerance: float
            Tolerance for cutpoint generation. A cutpoint will only be generated 
            between two values if they differ by tat least this value. (Default = 0.0)

        purity: float
            Minimum purity requirement for rules. This is an upper bound on the 
            number of points from any another class that are covered by a rule 
            (as a percentage of the total number of points covered by the rule).
            (Default = 0.95)

        mode: str
            The algorithm mode used for generating classsification rules. Possible
            values: {eager, lazy}
    '''

    def __init__(self, tolerance=0.0, purity=0.95, mode="eager"):
        self.tolerance = tolerance
        self.purity = purity
        self.mode = mode

        self.model = None

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
            self.model = MaxPatterns(self.purity)

        elif self.mode == 'lazy':
            self.model = LazyPatterns(self.purity)
        
        self.model.fit(Xbin, y)
        self.model.adjust(cpb, gsc)

        return self  # `fit` should always return `self`

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        return self.model.predict(X)