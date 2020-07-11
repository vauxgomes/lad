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

        att = -1
        for row in X.T:
            att += 1
            labels = None  # Previuos labels
            u = -9999  # Previuos xi

            # Finding transitions
            for v in sorted(np.unique(row)):
                variation = v - u # Current - Previous

                # Classes where v appears
                indexes = np.where(row == v)[0]
                __labels = set(y[indexes])

                # Main condition
                if labels is not None and variation > self.__tolerance:

                    # Testing for transition
                    if (len(labels) > 1 or len(__labels) > 1) or labels != __labels:
                        cid = len(self.__cutpoints)
                        self.__cutpoints[cid] = (att, u + variation/2.0)

                labels = __labels
                u = v

        return self.__cutpoints

    def transform(self, X):
        Xbin = np.empty((X.shape[0], 0), bool)

        for att, cutpoint in self.__cutpoints.values():
            # Binarizing
            row = X.T[att]
            row = row.reshape(X.shape[0], 1) <= cutpoint
            Xbin = np.hstack((Xbin, row))

        return Xbin

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)