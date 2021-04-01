import numpy as np
import pandas as pd
import datasets as dt # https://github.com/vauxgomes/datasets

from lad.lad import LADClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_validate

# Load
df = dt.load_iris()

X = df[df.columns[:-1]]
y = df[df.columns[-1]]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# Clasisfier
clf = LADClassifier(mode='eager')
clf.fit(X_train, y_train)

y_hat = clf.predict(X_test)

print(classification_report(y_test, y_hat))

print(clf)

# scores = cross_validate(LADClassifier(mode='eager'), X, y, scoring=['accuracy'])

# print(np.mean(scores['test_accuracy']))