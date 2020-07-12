![Header](img/header.png)

# Summary
 - [Project](#project)
   - [Description](#description)
   - [Related publications](#related-publications)
 - [Example](#example)
 - [Versions and tags](#versions-and-tags)

# Project

This is an open source implementation of the Logical Analysis of Data (LAD) Algorithm.

### Description

Logical Analysis of Data (LAD) is a rule-based machine learning algorithm based on ideas from Optimization and Boolean Function Theory. The LAD methodology was originally conceived by Peter L. Hammer, from Rutgers University, and has been described and developed in a number of papers since the late 80's. It has also been applied to classification problems arising in areas such as Medicine, Economics, and Bioinformatics. A list with representative publications about LAD will be made available here shortly.

LAD algorithm consists of dectecting hidden patterns capable of distinguishing observations in one class from all the other observations. The patterns are human readable which are used for reasoning of the decisions made by the classifier. 

> A Java implementation of a binary LAD classifier can be found [here](https://lia.ufc.br/~tiberius/lad/downloads.htm).

### Related publications
**An Implementation of Logical Analysis of Data**. Boros, E., P.L. Hammer, T. Ibaraki, A. Kogan, E. Mayoraz, I. Muchnik. IEEE Transactions on Knowledge and Data Engineering, vol 12(2), 292-306, 2000. (Link)[https://ieeexplore.ieee.org/abstract/document/842268?casa_token=y2NyWCbn7SsAAAAA:LCrKLdntpx-5GRNVdtU4F-Cnfs4VqsfWZTspa_yvgy_acfHvZjoZt_ZXKtHAOdiZGioUiNAN6m4FwQ]

**Maximum Patterns in Datasets**. Bonates, T.O., P.L. Hammer, A. Kogan. Discrete Applied Mathematics, vol. 156(6), 846-861, 2008. (Link)[https://www.sciencedirect.com/science/article/pii/S0166218X07002089]

**Classificação Supervisionada de Dados via Otimização e Funções Booleanas**. Gomes, V.S.D., T. O. Bonates. Anais do II Workshop Técnico-Científico de Computação, p.21-27, Mossoró, RN, Brazil, 2011.

# Example
As the code was implemented following sklean's classifiers format, its usage is quitte straightforward. See the code below.

```py
from lad import LADClassifier

from sklearn import datasets
from sklearn.model_selection import cross_val_score

# Dataset
X, y = datasets.load_iris(return_X_y=True)

# Classifier
lad = LADClassifier()

# CV
scores = cross_val_score(lad, X, y, cv=10, scoring="accuracy")
```

> The current version of lad doesn't implement a score function!

# Versions and tags

| Tag | Description | Algorithms | Status |
| -- | --  | -- | -- |
| v0.1 | Uses pandas for processing the data and build decision rules. | MaxPatterns | Published |
| v0.2 | Uses numpy instead of pandas. | MaxPatterns | Published |
| v0.3 | Fully validated code | MaxPatterns | Working |
| v0.4 | The Random Rule Generator | MaxPatterns, RandomRules | -- |
| v0.5 | Fully documented code | MaxPatterns, RandomRules | -- |
| v0.6 | The LAD lazy mode | MaxPatterns, RandomRules | -- |