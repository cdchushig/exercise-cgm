import numpy as np
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score
from sklearn.metrics import confusion_matrix, classification_report, f1_score


def get_metric_classification(scoring_metric, n_classes, average='macro'):

    if n_classes == 2:
        return scoring_metric
    else:
        if scoring_metric == 'roc_auc':
            return '{}_{}'.format(scoring_metric, 'ovo')
        elif scoring_metric == 'f1':
            return '{}_{}'.format(scoring_metric, average)


def compute_classification_prestations(y_true: np.array,
                                       y_pred: np.array,
                                       class_names: np.array,
                                       verbose=False,
                                       save_confusion_matrix=False
                                       ) -> (float, float, float, float):

    if len(np.unique(y_pred)) != len(class_names):
        y_pred = np.where(y_pred >= 0.5, 1.0, y_pred)
        y_pred = np.where(y_pred < 0.5, 0.0, y_pred)

    if verbose:
        print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)

    # if save_confusion_matrix:
    #     plot_confusion_matrix(cm, class_names, save_confusion_matrix)

    n_classes = len(set(y_true))

    return {'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'specificity': specificity_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
    }


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity