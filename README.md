# Project
Open source implementation of the Logical Analysis of Data (LAD) Algorithm.

## Description
LAD algorithm consists of dectecting hidden patterns capable of distinguishing observations in one class from all the other observations. The patterns are human readable which are used for reasoning of the decisions made by the classifier. 

This project is based on the works of T.O. Bonates, Peter L. Hammer and A. Kogan in [Maximum patterns in datasets](https://www.sciencedirect.com/science/article/pii/S0166218X07002089). A Java implementation of LAD can be found [here](https://lia.ufc.br/~tiberius/lad/downloads.htm).

## Branching

 - Master: Stable version (tagged)
 - Release: Next iteration
 - Feature: Implementation tasks
 - Bugfix

All but `master` and `release` branches must be deleted after usage.

## Directory structure

```
Project
├── lad
│   ├── binarizer.py
│   ├── lad.py
│   └── setcover.py
│
├── test
│   ├── test_binarizer.py
│   └── test_setcover.py
│
└── data
    ├── sample_bin.csv
    └── sample.csv
```
