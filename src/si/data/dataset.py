import numpy as np
import pandas as pd

from typing import Tuple, Sequence

class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None):
        """
        Dataset represents a machine learning tabular dataset

        :param X: numpy.ndarray (n_samples, n_features)
            The feature matrix
        :param y: numpy.ndarray (n_samples, 1)
            The label vector
        :param features: list of str (n_features)
            The features names
        :param label: str(1)
            The label name
        """
        if X is None:
            raise ValueError("X cannot be None")

        if features is None:
            features = [str(i) for i in range(X.shape[1])]
        else:
            features = list(features)

        if y is not None and label is None:
            label = "y"

        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the dataset

        :return: tuple (n_samples, n_features)
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Returns True if the dataset has a label

        :return: bool
        """
        return self.y is not None

    def get_classes(self) -> np.ndarray:
        """
        Returns the unique classes in the dataset
        :return: numpy.ndarray (n_classes)
        """
        #np.unique
        if self.y is None:
            raise ValueError("Dataset does not have a label")
        return np.unique(self.y)

    def get_mean(self) -> np.ndarray:
        """
        Returns the mean of each feature
        :return: numpy.ndarray (n_features)
        """
        return np.mean(self.X, axis=0) #axis tem de ser 0

    def get_variance(self) -> np.ndarray:
        """
        Returns the variance of each feature
        :return: numpy.ndarray (n_features)
        """
        return np.var(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        """
        Returns the median of each feature
        :return: numpy.ndarray (n_features)
        """
        return np.median(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        """
        Returns the minimum of each feature
        :return:
        """
        return np.min(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        """
        Returns the maximum of each feature
        :return: numpy.ndarray (n_features)
        """
        return np.max(self.X, axis=0)

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the dataset
        :return: pandas.DataFrame(n_features,5)
        """
        data = {
            "mean": self.get_mean(),
            "median": self.get_median(),
            "min": self.get_min(),
            "max": self.get_max(),
            "var": self.get_variance()
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)


    #exercício 2:
    def dropna(self):
        """
         Removes all samples that contain at least one null value (NaN)
        """
        self.X = self.X[~np.isnan(self.X).any(axis=1)]
        return self.X

    def fillna(self, n:int):
        """
        replaces all null values with another value (function/method argument)
        """
        self.X = np.nan_to_num(self.X, nan=n)

    def from_random(n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Criar um dataset a partir de dados aleatórios

        :param n_samples: int
            O número de samples
        :param n_features: int
            O número de features
        :param n_classes: int
            O número de classes
        :param features: list of str
            O nome de features
        :param label: str
            O nome de label

        :return:
            Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return Dataset(X, y, features=features, label=label)

    def from_dataframe(df: pd.DataFrame, label: str = None):
        """
        Cria um Dataset a partir de um Dataframe (pandas)
        :param df: pandas.DataFrame
            O DataFrame
        :param label: str
            O nome da label
        :return:
            Dataset
        """
        if label:
            X = df.drop(label, axis=1).to_numpy()
            y = df[label].to_numpy()
        else:
            X = df.to_numpy()
            y = None

        features = df.columns.tolist()
        return Dataset(X, y, features=features, label=label)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converte o dataset para um DataFrame (pandas)

        :return:
            pandas.DataFrame
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df


if __name__ == '__main__':
    x = np.array([[1, 2, 3], [1, 2, 3]])
    y = np.array([1, 2])
    y_a= np.array([1,2,3,4,5])
    features = ['A', 'B', 'C']
    label = 'y'
    dataset = Dataset(X=x, y=y, features=features, label=label)
    print(dataset.shape()) #tem 2 classes
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.get_mean())
    print(dataset.get_variance())
    print(dataset.get_median())
    print(dataset.get_median())
    print(dataset.get_min())
    print(dataset.get_max())
    print(dataset.summary())



