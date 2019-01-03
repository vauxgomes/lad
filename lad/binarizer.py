#!/usr/bin/env python

"""
	There is a cut point if in the sorted list of values of a given feature
	has two values of different classes and the absolute difference between
	those values is bigger then the chosen tolerance
"""		

# Imports
import pandas as pd

# Docs
__author__ = 'Vaux Gomes'
__version__ = '1.0.0'

#
class CutPoint(object):
	""" Cut Point Binarizer """

	#
	def __init__(self, tolerance = 0.0):
		self.tolerance = tolerance

	#
	def fit(self, df, columns, target):
		df_ = pd.DataFrame(columns=[])

		for col in columns:
			feature = df[[col, target]].sort_values(by=col)
			groups = df[[col, target]].groupby(col)
			
			feature['Shifted'] = feature[col].shift(-1)
			feature['Shifted'].iloc[-1] = feature[col].iloc[-1]

			feature['Diff'] = (feature[col] - feature['Shifted']).abs()
			feature['CutPoint'] = feature[col] + (feature['Diff'] / 2.0)

			feature['Transaction'] = feature[[col, 'Shifted']].apply(
				lambda x: len(set(groups.get_group(x[0])['Class']) ^
					set(groups.get_group(x[1])['Class'])) > 0, axis=1)
			feature['Transaction'] = feature['Transaction'] & (feature['Diff'] > self.tolerance)

			for i, transaction in enumerate(feature['Transaction']):
				if transaction:
					df_['{}_{}'.format(col, i)] = feature['CutPoint'].iloc[i] > feature[col]
			
		return df_.join(df[[target]])