from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import numpy as np


class XGBWithThreshold(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = threshold  # Adiciona o limiar como hiperparâmetro
        self.model = xgb.XGBClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X):
        proba = self.model.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        params = self.model.get_params(deep=deep)
        params['threshold'] = self.threshold
        return params

    def set_params(self, **params):
        if 'threshold' in params:
            self.threshold = params.pop('threshold')  # Extrai threshold se existir
        self.model.set_params(**params)  # Define os parâmetros restantes no modelo XGBClassifier
        return self


class LogRegWithThreshold(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = threshold  # Adiciona o limiar como hiperparâmetro
        self.model = LogisticRegression(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_

        # A regressão logística não tem feature_importances_, mas podemos usar coef_
        self.feature_importances_ = np.abs(self.model.coef_).flatten()
        return self

    def predict(self, X):
        proba = self.model.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        params = self.model.get_params(deep=deep)
        params['threshold'] = self.threshold
        return params

    def set_params(self, **params):
        if 'threshold' in params:
            self.threshold = params.pop('threshold')  # Extrai threshold se existir
        self.model.set_params(**params)  # Define os parâmetros restantes no modelo
        return self


class TreeWithThreshold(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = threshold  # Adiciona o limiar como hiperparâmetro
        self.model = DecisionTreeClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X):
        proba = self.model.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        params = self.model.get_params(deep=deep)
        params['threshold'] = self.threshold
        return params

    def set_params(self, **params):
        if 'threshold' in params:
            self.threshold = params.pop('threshold')  # Extrai threshold se existir
        self.model.set_params(**params)  # Define os parâmetros restantes no modelo XGBClassifier
        return self

class CascadedXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, primary_params=None, secondary_params=None):
        """
        Modelo cascata que combina dois classificadores XGBoost.

        Parameters:
        primary_params (dict): Hiperparâmetros para o modelo primário (detecção de ataque).
        secondary_params (dict): Hiperparâmetros para o modelo secundário (classificação do ataque).
        """
        self.primary_params = primary_params or {}
        self.secondary_params = secondary_params or {}

        # Criando os modelos XGBoost
        self.primary_model = xgb.XGBClassifier(**self.primary_params)
        self.secondary_model = xgb.XGBClassifier(**self.secondary_params)

        self.classes_ = None

    def fit(self, X, y_primary, y_secondary):
        """
        Treina o modelo em duas etapas:
        1. O modelo primário (normal vs ataque)
        2. O modelo secundário (classificação do ataque)

        Parameters:
        X (array-like): Features de entrada.
        y_primary (array-like): Labels binários (0 para normal, 1 para ataque).
        y_secondary (array-like): Labels multiclasses para ataques.

        Returns:
        self
        """
        # Treinar o modelo primário (normal vs ataque)
        self.primary_model.fit(X, y_primary)

        # Obtendo o threshold do modelo primário
        threshold = self.primary_model.get_params().get('threshold', 0.5)

        # Identificar amostras classificadas como ataque
        proba_primary = self.primary_model.predict_proba(X)[:, 1]
        ataque_indices = proba_primary >= threshold

        X_attack = X[ataque_indices]
        y_attack = y_secondary[ataque_indices]

        # Treinar o modelo secundário (classificação de ataque)
        self.secondary_model.fit(X_attack, y_attack)

        # Definir classes do modelo (normal + classes de ataque)
        self.classes_ = np.concatenate(([0], np.unique(y_attack)))

        return self

    def predict(self, X):
        """
        Realiza a predição cascata.

        Parameters:
        X (array-like): Features de entrada.

        Returns:
        array-like: Predições (0 para normal ou uma das 9 classes de ataque).
        """
        # Predição do modelo primário
        proba_primary = self.primary_model.predict_proba(X)[:, 1]

        # Obtendo threshold do modelo primário
        threshold = self.primary_model.get_params().get('threshold', 0.5)

        predictions = np.zeros(X.shape[0], dtype=int)
        ataque_indices = proba_primary >= threshold

        # Se houver ataques detectados, classificar com o modelo secundário
        if np.any(ataque_indices):
            X_attack = X[ataque_indices]
            predictions_attack = self.secondary_model.predict(X_attack)
            predictions[ataque_indices] = predictions_attack

        return predictions

    def predict_proba(self, X):
        """
        Retorna as probabilidades de predição para cada classe.

        Parameters:
        X (array-like): Features de entrada.

        Returns:
        array-like: Probabilidades para cada uma das 10 classes.
        """
        proba_primary = self.primary_model.predict_proba(X)[:, 1]
        proba_final = np.zeros((X.shape[0], len(self.classes_)))

        # Considera 'normal' como classe 0 (probabilidade inversa ao ataque)
        proba_final[:, 0] = 1 - proba_primary  # Probabilidade de ser 'normal'

        threshold = self.primary_model.get_params().get('threshold', 0.5)
        ataque_indices = proba_primary >= threshold

        if np.any(ataque_indices):
            X_attack = X[ataque_indices]
            proba_secondary = self.secondary_model.predict_proba(X_attack)

            # Inserir as probabilidades das classes de ataque nas posições corretas
            proba_final[ataque_indices, 1:] = proba_secondary

        return proba_final

    def get_params(self, deep=True):
        """
        Retorna os hiperparâmetros do modelo.

        Returns:
        dict: Parâmetros dos modelos e threshold.
        """
        params = {
            "primary_params": self.primary_model.get_params(deep),
            "secondary_params": self.secondary_model.get_params(deep),
        }
        return params

    def set_params(self, **params):
        """
        Define os hiperparâmetros do modelo.

        Parameters:
        params (dict): Parâmetros para atualizar nos modelos.

        Returns:
        self
        """
        if 'primary_params' in params:
            self.primary_model.set_params(**params.pop('primary_params'))

        if 'secondary_params' in params:
            self.secondary_model.set_params(**params.pop('secondary_params'))

        return self



class CascadedLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, primary_params=None, secondary_params=None):
        """
        Modelo cascata que combina dois classificadores LogisticRegression.

        Parameters:
        primary_params (dict): Hiperparâmetros para o modelo primário (detecção de ataque).
        secondary_params (dict): Hiperparâmetros para o modelo secundário (classificação do ataque).
        """
        self.primary_params = primary_params or {}
        self.secondary_params = secondary_params or {}

        # Criando os modelos XGBoost
        self.primary_model = LogisticRegression(**self.primary_params)
        self.secondary_model = LogisticRegression(**self.secondary_params)

        self.classes_ = None

    def fit(self, X, y_primary, y_secondary):
        """
        Treina o modelo em duas etapas:
        1. O modelo primário (normal vs ataque)
        2. O modelo secundário (classificação do ataque)

        Parameters:
        X (array-like): Features de entrada.
        y_primary (array-like): Labels binários (0 para normal, 1 para ataque).
        y_secondary (array-like): Labels multiclasses para ataques.

        Returns:
        self
        """
        # Treinar o modelo primário (normal vs ataque)
        self.primary_model.fit(X, y_primary)

        # Obtendo o threshold do modelo primário
        threshold = self.primary_model.get_params().get('threshold', 0.5)

        # Identificar amostras classificadas como ataque
        proba_primary = self.primary_model.predict_proba(X)[:, 1]
        ataque_indices = proba_primary >= threshold

        X_attack = X[ataque_indices]
        y_attack = y_secondary[ataque_indices]

        # Treinar o modelo secundário (classificação de ataque)
        self.secondary_model.fit(X_attack, y_attack)

        # Definir classes do modelo (normal + classes de ataque)
        self.classes_ = np.concatenate(([0], np.unique(y_attack)))

        return self

    def predict(self, X):
        """
        Realiza a predição cascata.

        Parameters:
        X (array-like): Features de entrada.

        Returns:
        array-like: Predições (0 para normal ou uma das 9 classes de ataque).
        """
        # Predição do modelo primário
        proba_primary = self.primary_model.predict_proba(X)[:, 1]

        # Obtendo threshold do modelo primário
        threshold = self.primary_model.get_params().get('threshold', 0.5)

        predictions = np.zeros(X.shape[0], dtype=int)
        ataque_indices = proba_primary >= threshold

        # Se houver ataques detectados, classificar com o modelo secundário
        if np.any(ataque_indices):
            X_attack = X[ataque_indices]
            predictions_attack = self.secondary_model.predict(X_attack)
            predictions[ataque_indices] = predictions_attack

        return predictions

    def predict_proba(self, X):
        """
        Retorna as probabilidades de predição para cada classe.

        Parameters:
        X (array-like): Features de entrada.

        Returns:
        array-like: Probabilidades para cada uma das 10 classes.
        """
        proba_primary = self.primary_model.predict_proba(X)[:, 1]
        proba_final = np.zeros((X.shape[0], len(self.classes_)))

        # Considera 'normal' como classe 0 (probabilidade inversa ao ataque)
        proba_final[:, 0] = 1 - proba_primary  # Probabilidade de ser 'normal'

        threshold = self.primary_model.get_params().get('threshold', 0.5)
        ataque_indices = proba_primary >= threshold

        if np.any(ataque_indices):
            X_attack = X[ataque_indices]
            proba_secondary = self.secondary_model.predict_proba(X_attack)

            # Inserir as probabilidades das classes de ataque nas posições corretas
            proba_final[ataque_indices, 1:] = proba_secondary

        return proba_final

    def get_params(self, deep=True):
        """
        Retorna os hiperparâmetros do modelo.

        Returns:
        dict: Parâmetros dos modelos e threshold.
        """
        params = {
            "primary_params": self.primary_model.get_params(deep),
            "secondary_params": self.secondary_model.get_params(deep),
        }
        return params

    def set_params(self, **params):
        """
        Define os hiperparâmetros do modelo.

        Parameters:
        params (dict): Parâmetros para atualizar nos modelos.

        Returns:
        self
        """
        if 'primary_params' in params:
            self.primary_model.set_params(**params.pop('primary_params'))

        if 'secondary_params' in params:
            self.secondary_model.set_params(**params.pop('secondary_params'))

        return self