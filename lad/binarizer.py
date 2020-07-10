#!/usr/bin/env python
import pandas as pd


class CutpointBinarizer():

    def __init__(self, tolerance=0.0):
        self.__tolerance = tolerance
        self.__cutpoints = {}
        self.__size = 0

    def get_cutpoints(self):
        return self.__cutpoints

    def fit(self, X, y):
        self.__cutpoints.clear()

        for att in X.columns:

            labels = None  # Previuos labels
            u = -9999  # Previuos xi

            # Finding transitions
            for v in sorted(X[att].unique()):
                variation = v - u  # Current - Previous

                # Class where v appears
                indexes = X[X[att] == v].index
                __labels = set(y[indexes].unique())

                # Main condition
                if labels is not None and variation > self.__tolerance:

                    # Testing for transition
                    if (len(labels) > 1 or len(__labels) > 1) or labels != __labels:
                        cid = f'C{len(self.__cutpoints)}'
                        self.__cutpoints[cid] = (att, u + variation/2.0)

                labels = __labels
                u = v

        return self.__cutpoints

    def transform(self, X):
        Xbin = pd.DataFrame()

        for b, cutpoint in self.__cutpoints.items():
            att, v = cutpoint

            # Binarizing
            Xbin[b] = X[att] <= v

        return Xbin

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
