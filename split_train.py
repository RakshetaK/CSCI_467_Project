import numpy as np 
from sklearn.model_selection import train_test_split

def split_train():
    data = np.loadtxt("dataset_467/optdigits.tra", delimiter=",")

    X = data[:,:-1]
    y = data[:, -1]

    X_train, X_dev, y_train, y_dev = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_dev, y_train, y_dev

