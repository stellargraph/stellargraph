from sklearn import metrics
import numpy as np


def calc_f1(y_true, y_pred, sigmoid):
    if not sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

    return (metrics.f1_score(y_true, y_pred, average='micro'),
            metrics.f1_score(y_true, y_pred, average='macro'))
