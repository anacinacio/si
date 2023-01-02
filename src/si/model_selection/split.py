from typing import Tuple
from si.data.dataset import Dataset
import numpy as np

def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42)->tuple:
    """
    - Gera permutações usando o np.random.permutation
    - Infere o número de amostras no dataset de teste e treino
    - Seleciona treino e teste usando as permutações

    :param dataset: Dataset
        dataset para dividir em treino e teste

    :param test_size: float
        tamanho do dataset de test (e.g.,20%)

    :param random_state: int
        seed para gerar permutações

    :return:
        Um tuplo com dataset de treino e dataset de teste

    """
    np.random.seed(random_state)

    n_samples = dataset.shape()[0]
    n_test = int(n_samples * test_size)
    permutations = np.random.permutation(n_samples)

    test_idxs = permutations[:n_test]

    train_idxs = permutations[n_test:]

    train = Dataset(dataset.X[train_idxs],
                    dataset.y[train_idxs],
                    features=dataset.features,
                    label=dataset.label)

    test = Dataset(dataset.X[test_idxs],
                   dataset.y[test_idxs],
                   features=dataset.features,
                   label=dataset.label)
    return train, test