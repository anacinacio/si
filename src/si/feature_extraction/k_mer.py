import itertools

import numpy as np

from si.data.dataset import Dataset


class KMer:
    """
    Um descritor de sequência que devolve a composição k-mer da sequência

    :parameter
    k : int
        tamanho da substring

    alphabet: str
        onde o utilizador fornece o alfabeto da sequência biológica

    :attributes
    k_mers : list of str
        todos os k-mers possíveis
    """
    def __init__(self, k: int = 3, alphabet: str = 'DNA'):
        """
        :parameter
        k : int
            tamanho da substring

        alphabet: str
            onde o utilizador fornecef o alabeto da sequência biológica
        :attributes
        k_mers : list of str
            todos os k-mers possíveis.
        """
        # parametros
        self.k = k
        self.alphabet = alphabet.upper()

        #casos possiveis do alfabeto da sequencia biológica
        if self.alphabet == 'DNA':
            #se for nucleotidos
            self.alphabet = 'ACTG'
        elif self.alphabet == 'PROT':
            #se for aminoacidos
            self.alphabet = 'FLIMVSPTAY_HQNKDECWRG'
        else:
            self.alphabet = self.alphabet

        # atributos
        self.k_mers = None

    def fit(self, dataset: Dataset) -> 'KMer':
        """
        estima todos os k k-mers possíveis

        :param
        dataset : Dataset

        :return
        self: KMer
            k-mers possíveis
        """
        # gera os k-mers
        self.k_mers = [''.join(k_mer) for k_mer in itertools.product(self.alphabet, repeat=self.k)]
        return self

    def _get_sequence_k_mer_composition(self, sequence: str) -> np.ndarray:
        """
        Calcula a composição k-mer da sequencia

        :param
        sequence : str
            A sequencia para calcular a composição de k-mer

        :return
        List: float
            A composição de k-mer da sequencia
        """
        #calcula a composição de k-mer
        counts = {k_mer: 0 for k_mer in self.k_mers}

        for i in range(len(sequence) - self.k + 1):
            #k_mer = sequence[i:i + self.k]
            #counts[k_mer] += 1
            counts[sequence[i:i+self.k]]+=1
        #normaliza as contagens
        return np.array([counts[k_mer] / len(sequence) for k_mer in self.k_mers])

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transforma o dataset
        calcula a frequência normalizada de cada k-mer em cada sequência
        :param
        dataset : Dataset
            O dataset

        :return
        Dataset
            dataset transformado
        """
        #calcula a composição de k-mer
        sequences_k_mer_composition = [self._get_sequence_k_mer_composition(sequence) for sequence in dataset.X[:, 0]]
        sequences_k_mer_composition = np.array(sequences_k_mer_composition)

        #cria um novo dataset
        return Dataset(X=sequences_k_mer_composition, y=dataset.y, features=self.k_mers, label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
       Corre o fit e depois o transform, ou seja, ajustar-se aos dados, depois transformá-los.

        :param dataset: Dataset
            Dataset object
        :return: np.darray
            dataset transformado
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':
    from si.data.dataset import Dataset

    dataset_ = Dataset(X=np.array([['ACTGTTTAGCGGA', 'ACTGTTTAGCGGA']]),
                       y=np.array([1, 0]),
                       features=['sequence'],
                       label='label')

    k_mer_ = KMer(k=2)
    dataset_ = k_mer_.fit_transform(dataset_)
    print(dataset_.X)
    print(dataset_.features)
