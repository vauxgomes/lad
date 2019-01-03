#!/usr/bin/env python

"""
	Classic Set Cover Problem
"""		

# Imports
import pandas as pd
import numpy as np

# Docs
__author__ = 'Vaux Gomes'
__version__ = '1.0.0'

#
class GreedySetCover(object):
	""" General Greedy SetCover Algorithm """

	#
	def __init__(self):
		super(GreedySetCover, self).__init__()

	#
	def fit(self, df):
		columns = list()

		while len(df) != 0:
			pcount = df.sum(axis=0)
			ncount = abs(pcount - len(df))

			pindex = pcount.idxmax()
			nindex = ncount.idxmax()

			if pcount[pindex] >= ncount[nindex]:
				selected = pindex
				selected_val = True
			else:
				selected = nindex
				selected_val = False

			df.drop(df[df[selected] == selected_val].index, axis=0, inplace=True)
			df.drop([selected], axis=1, inplace=True)

			columns.append(selected)

		return columns

#
class RandomSetCover(object):
	""" Random SetCover Algorithm """

	#
	def __init__(self, seed = 2):
		self.seed = seed

	#
	def fit(self, df, sample_size = 1):
		np.random.seed(self.seed)
		columns = list()

		if sample_size <= 0:
			sample_size = len(df.columns)

		while len(df) != 0:
			col = df[np.random.choice(df.columns, size=sample_size)].sum(axis=0).idxmax()
			columns.append(col)

			df = df[df[col] == 0]
			df = df.drop([col], axis=1)

		return columns