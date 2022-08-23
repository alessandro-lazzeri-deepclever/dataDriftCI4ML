from sklearn.datasets import make_classification
from pickle import dump, load
from random import random


def create_dataset():
    datasets = make_classification(n_samples=50000, n_features=20, n_informative=15, n_classes=5, random_state=666)
    dump(datasets, open('dataset/datasets_v001.pkl', 'wb'))

def drift(X,y, n = 1):

    drift = 1.05
    drift_col_chance = 0.5

    for v in range(2,n+2):
        for i in range(X.shape[1]):

            if random() > drift_col_chance:
                X[:,i] *= drift

        dump((X,y), open('dataset/datasets_v%03d.pkl' % v, 'wb'))

if __name__ == '__main__':
    create_dataset()

    X, y = load(open('dataset/datasets_v%03d.pkl' % 1, 'rb'))
    n = 10
    drift(X,y,n)

