import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier:
    """
    Ensemble classifier that uses the majority vote to predict the class labels.

    :parameter
    models: array-like, shape = [n_models]
        conjunto de modelos

    final_model:
        o modelo final
    """

    def __init__(self, models, final_model):
        """
        Inicializar o classificador do ensemble

        :param
        models: array-like, shape = [n_models]
            modelos diferentes para o ensemble.
        """
        #parametros
        self.models = models
        self.final_model = final_model

    def fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        treina os modelos do ensemble

        :parameter
        dataset : Dataset
            Training data

        :return
        self : VotingClassifier
            The fitted model
        """

        #Treina o conjunto de modelos
        for model in self.models:
            model.fit(dataset)

        #Obtém previsões de cada modelo treinado anteriormente
        predicitions = []
        for model in self.models:
            predicitions.append(model.predict(dataset))

        #Treina o modelo final usando as previsões obtidas anteriormente
        self.final_model.fit(Dataset(dataset.X, np.array(predicitions).T))

        #Retorna ele próprio
        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        estima a variável de saída usando os modelos treinados e o modelo final

        :parameter
        dataset : Dataset
            Test data
        :return
        y : array-like, shape = [n_samples]
            The predicted class labels
        """
        #Obtém previsões de cada modelo no conjunto de modelos
        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))

        #Obtém as previsões finais usando o modelo final e as previsões obtidas anteriormente

        predictions_f = self.final_model.predict(Dataset(dataset.X, np.array(predictions).T))
        return predictions_f

    def score(self, dataset: Dataset) -> float:
        """
        calcula o erro entre as previsões e os valores reais

        :param
        dataset : Dataset
            The test data.

        :return
        score : float
            Mean accuracy
        """
        #Estima os valores de Y usando os modelos treinados e o modelo final
        predictions_f = self.predict(dataset)

        #Calcula a accuracy entre os valores reais e as previsões
        return accuracy(dataset.y, predictions_f)