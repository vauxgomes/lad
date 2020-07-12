# Project
Open source implementation of the Logical Analysis of Data (LAD) Algorithm.

## Description
LAD algorithm consists of dectecting hidden patterns capable of distinguishing observations in one class from all the other observations. The patterns are human readable which are used for reasoning of the decisions made by the classifier. 

This project is based on the works of T.O. Bonates, Peter L. Hammer and A. Kogan in [Maximum patterns in datasets](https://www.sciencedirect.com/science/article/pii/S0166218X07002089). A Java implementation of LAD can be found [here](https://lia.ufc.br/~tiberius/lad/downloads.htm).

## Instalation

#### By cloning the repository
To install this Python package, you need to clone the repository.

```sh
$ git clone https://github.com/vauxgomes/lad-classifier.git
```

Then just run the setup.py file from that directory.

```sh
$ sudo python setup.py install
```

#### Or using pip

```sh
$ python -m pip install git+https://github.com/vauxgomes/lad-classifier.git#egg=lad-classifier
```

## Versions and tags

| Tag | Description | Version | Status |
| -- | --  | -- | -- |
| v0.1 | Uses pandas for processing the data and build decision rules. | MaxPatterns | Published |
| v0.2 | Uses numpy instead of pandas. | MaxPatterns | Published |
| v0.3 | Fully validated code | MaxPatterns | Coding |
| v0.4 | The Random Rule Generator | MaxPatterns, RandomRules | -- |
| v0.5 | Fully documented code | MaxPatterns, RandomRules | -- |
| v0.6 | The LAD lazy mode | MaxPatterns, RandomRules | -- |