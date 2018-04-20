from sklearn import metrics


def calc_f1(y_true, y_pred):
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    return (metrics.f1_score(y_true, y_pred, average='micro'),
            metrics.f1_score(y_true, y_pred, average='macro'))
