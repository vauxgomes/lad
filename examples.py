import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate
from lad.lad import LADClassifier

def main():
    data = 'bcw'
    df = None

    if data == 'iris':
        df = pd.read_csv('data/iris.data',
                         names='att0 att1 att2 att3 class'.split())
    elif data == 'bcw':
        df = pd.read_csv(
            'data/bcw.data', names='id att0 att1 att2 att3 att4 att5 att6 att7 att8 class'.split())
        df['att5'] = df['att6'].replace('?', np.NaN).astype(int)
        df = df.drop(columns=['id'])
        df = df.dropna()

    # Train
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Classifier
    clf = LADClassifier()

    # CV
    cv = 5
    scores = cross_validate(clf, X, y, cv=cv, scoring=(
        'accuracy', 'precision_weighted', 'f1_weighted'))

    #
    print_scores(scores, cv)


def print_scores(scores, cv):
    print('-'*67)
    print(' | '.join([k.replace('test_', '').center(max(8, len(k) - (5 if k.startswith('test_') else 0)))
                      for k in scores.keys()]))
    print('-'*67)

    for i in range(cv):
        print(' | '.join([f'{scores[k][i]:.4}'.center(
            max(8, len(k) - (5 if k.startswith('test_') else 0))) for k in scores.keys()]))

    print('-'*67)
    print(' | '.join([f'{np.average(scores[k]):.4}'.center(
        max(8, len(k) - (5 if k.startswith('test_') else 0))) for k in scores.keys()]))
    print('-'*67)

if __name__ == '__main__':
    main()