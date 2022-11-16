from typing import Type, Union
import numpy as np
import pandas as pd

from si.data.dataset import Dataset

def read_csv (filename: str, sep: str =',', features: bool, label: bool=False) -> Dataset:
    """
    Read a csv file (data file) into a Dataset object
    :param filename: str, path to the file
    :param sep: str, optional
        The separator used in the file, by default False
    :param features: bool, optional
        Whether the file has a header, by default False
    :param label: bool, optional
        Whether the file has a label, by default False
    :return: Dataset
        The dataset object
    """
    if features and label:
        features = data.columns[:-1]
        label = data.columns[-1]
        X = data.iloc[:,:-1].to_numpy()
        y = data.iloc[:,-1].to_numpy()

    elif features and not label:
        features = data.columns
        X = data.to_numpy()
        y = None

    elif not features and label:
        X = data.iloc[:,:-1].to_numpy()
        y = data.iloc[:,-1].to_numpy()
        features = None
        label = None

    else:
        X = data.to_numpy()
        y = None
        features = None
        label = None

    return Dataset(X, y, features = features, label = label)



def write_csv (dataset: Dataset, filename: str, sep:str = ',', features: bool = False, label: bool = False)->None:
    """
    Writes a Dataset object to a csv file
    :param dataset: Dataset
    :param filename: str
    :param sep: str, optional
        The separator used in the file, by default ','
    :param features: bool, optional
        Whether the file has a header, by default False
    :param label: bool, optional
        Whether the file has a label, by default False
    """
    data = pd.DataFrame(dataset.X)

    if features:
        data.columns = dataset.features

    if label:
        data[dataset.label] = dataset.y

    data.to_csv(filename, sep = sep, index = False)











