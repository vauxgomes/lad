#!/usr/bin/env python

"""
	Template
	https://github.com/scikit-learn-contrib
"""

# Imports
import numpy 	
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

# Docs
__author__ = 'Vaux Gomes'
__version__ = '1.0.0'


class LADClassifier(BaseEstimator):
	"""  LAD Classifier """

	def __init__(self, tolearance=0.0, purity=0.95):
		"""
			LAD Classifier Constructor

			Parameter:
			tolerance -- Cutpoint tolerance. It must be bigger than or equal to zero
			purity -- Minimum rule purity
		"""

		self.tolearance = tolearance
		self.purity = purity
		#self.pattern_length = pattern_length
		#self.boost_rules = boost_rules

	def fit(self, X, y):
		X, y = check_X_y(X, y, accept_sparse=True)
		self.is_fitted_ = True

		# Binarization
		# Feature Selection
		
		return self # `fit` should always return `self`

	def predict(self, X):
		X = check_array(X, accept_sparse=True)
		check_is_fitted(self, 'is_fitted_')

		#return np.ones(X.shape[0], dtype=np.int64)

	#
	def calc_purity(class_, n_positive, n_negative):
		if class_:
			return float(n_positive) / (n_positive + n_negative)
		else:
			return float(n_negative) / (n_positive + n_negative)

	#
	def calc_discrepancy(instance, df):
		return (instance != df).values.sum()

	#
	def calc_coverage(instance, df):
		return (instance == df)