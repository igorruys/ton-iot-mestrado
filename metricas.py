from sklearn.metrics import fbeta_score
from numpy import sqrt

def custom_fbeta(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)

