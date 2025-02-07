from sklearn.metrics import fbeta_score

def custom_fbeta(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1.22)

