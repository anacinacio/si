import numpy as np

from typing import Dict, Tuple, Callable, Union, List
from si.data.dataset import Dataset
from si.model_selection.cross_validate import cross_validate

Num = Union[int, float]

def randomized_search_cv(model,
                   dataset: Dataset,
                   parameter_distribution: Dict[str, Tuple],
                   scoring: Callable = None,
                   cv: int = 5,
                   n_iter: int =10,
                   test_size: float = 0.2) -> Dict[str, Tuple[str, Num]]:
    """
    implementa uma estratégia de otimização de parâmetros de usando Nº
    combinações aleatórias.
    avalia apenas um conjunto aleatório de parâmetros retirados de uma distribuição ou conjunto de valores possíveis.

    :parameter
    model
        modelo a validar
    dataset: Dataset
        dataset de validação
    parameter_distribution: Dict[str, Tuple]
        os parâmetros para a procura. Dicionário com nome do parâmetro e distribuição de valores
    scoring: Callable
        função de score
    cv: int
        numero de folds.
    n_iter: int
        numero de combinações aleatórias de parametros
    test_size: float
        tamanho do dataset de teste

    :return
    scores: List[Dict[str, List[float]]]
        Uma lista de dicionários com a combinação dos parâmetros e os scores de treino e teste
    """
    #deve retornar uma lista de dicionários. Os dicionários devem conter as seguintes chaves:
    scores = {"parameters": [], "seeds": [], "train": [], "test": []}

    #verifica se os parâmetros fornecidos existem no modelo
    for param in parameter_distribution:
        if not hasattr(model, param):
            raise AttributeError(f"Model {model} does not have parameter {param}.")

    #n_iter combinações de parâmetros
    for i in range(n_iter):
        #random seed
        random_seed = np.random.randint(0, 1000)

        #armazenar seed
        scores["seeds"].append(random_seed)

        parameters = {}
        #definir parametros
        for param, value in parameter_distribution.items():
            #para retirar um valor aleatório da distribuição de valores de cada parâmetro.
            parameters[param] = np.random.choice(value)

        #parametros para o modelo
        for param, value in parameters.items():
            setattr(model, param, value)

        #cross_validation com a combinação
        score = cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv, test_size=test_size)

        #Guarda a combinação de parâmetros e os scores obtidos.
        scores["parameters"].append(parameters)
        scores["train"].append(score["train"])
        scores["test"].append(score["test"])

    return scores


if __name__ == '__main__':
    from io_folder.module_csv import read_csv
    from sklearn.preprocessing import StandardScaler
    from linear_model.logistic_regression import LogisticRegression

    breast_bin = read_csv('C:/Users/Carolina/Documents/GitHub/si/datasets/breast-bin.csv', sep=',', features=True,
                          label=True)
    breast_bin.X = StandardScaler().fit_transform(breast_bin.X)
    modelo_lg = LogisticRegression()

    param_distribution = {
        'l2_penalty': np.linspace(1, 10, 10),
        'alpha': np.linspace(0.001, 0.0001, 100),
        'max_iter': np.linspace(1000, 2000, 200, dtype=int)
    }

    # cross validate the model
    scores = randomized_search_cv(modelo_lg,
                                  breast_bin,
                                  parameter_distribution = param_distribution,
                                  cv=3, n_iter=10)

    print('Scores: \n', scores)