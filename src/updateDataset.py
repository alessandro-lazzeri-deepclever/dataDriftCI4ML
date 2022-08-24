from sklearn.datasets import make_classification
from pickle import dump, load
from random import random
import sys
import yaml
import os


def drift(X,y, n = 1):

    drift = 1.05
    drift_col_chance = 0.5

    for v in range(2,n+2):
        for i in range(X.shape[1]):

            if random() > drift_col_chance:
                X[:,i] *= drift

        dump((X,y), open('dataset/prepared/datasets_v%03d.pkl' % v, 'wb'))


if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython updateDataset.py n_datasets\n")
    sys.exit(1)

params = yaml.safe_load(open("params.yaml"))["dataset"]

os.makedirs(os.path.join("dataset", "original"), exist_ok=True)
os.makedirs(os.path.join("dataset", "prepared"), exist_ok=True)

X, y = load(open('dataset/original/datasets_v%03d.pkl' % 1, 'rb'))
n = int(sys.argv[1])
drift(X, y, n)

