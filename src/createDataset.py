from sklearn.datasets import make_classification
from pickle import dump, load
from random import random
import sys
import yaml
import os


def create_dataset(n_samples = 5000,n_features = 10):
    datasets = make_classification(n_samples, n_features, n_informative=15, n_classes=5, random_state=666)
    dump(datasets, open('dataset/original/datasets_v001.pkl', 'wb'))


if len(sys.argv) != 1:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython createDataset.py\n")
    sys.exit(1)

params = yaml.safe_load(open("params.yaml"))["dataset"]

os.makedirs(os.path.join("dataset", "original"), exist_ok=True)
os.makedirs(os.path.join("dataset", "prepared"), exist_ok=True)

create_dataset(params["n_samples"],params["n_features"])

#X, y = load(open('dataset/original/datasets_v%03d.pkl' % 1, 'rb'))
#n = int(sys.argv[1])
#drift(X, y, n)

