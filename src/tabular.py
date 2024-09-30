import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, LeaveOneOut, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
from tabpfn import TabPFNClassifier

from evaluator import compute_classification_prestations

PATH_PROJECT_DIR = Path(__file__).resolve().parents[1]
PATH_PROJECT_METRICS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'metrics')


def save_metrics(list_total_metrics: list):
    csv_pathfile = str(Path.joinpath(PATH_PROJECT_METRICS, 'metrics_classification.csv'))

    if os.path.exists(csv_pathfile):
        df_metrics = pd.read_csv(csv_pathfile)
    else:
        df_metrics = pd.DataFrame()

    for dict_metrics in list_total_metrics:
        df_metrics = pd.concat([df_metrics, pd.DataFrame([dict_metrics])], ignore_index=True)

    df_metrics.to_csv(csv_pathfile, index=False)


def get_clf_hyperparameters(classifier: str):
    selected_clf = None
    param_grid = {}

    if classifier == 'xgb':
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01, 0.001],
            'subsample': [0.5, 0.7, 1]
        }
        selected_clf = xgb.XGBClassifier()

    elif classifier == 'lgb':
        selected_clf = LGBMClassifier()
        param_grid = {
            'num_leaves': [10, 31, 127],
            'boosting_type': ['gbdt', 'rf'],
            'learning rate': [0.1, 0.001, 0.003]
        }
    elif classifier == 'rf':
        selected_clf = RandomForestClassifier()

        param_grid = {
            'n_estimators': [5, 10, 20],
            'max_depth': range(1, 16, 2),
        }
    elif classifier == 'tabpfn':
        selected_clf = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
        param_grid = {
            'N_ensemble_configurations': [16, 32]
        }

    return selected_clf, param_grid


def perform_clf(estimator_name: str,
                scoring_estimator,
                x_train: np.array,
                y_train: np.array,
                x_test: np.array,
                y_test: np.array,
                cv,
                n_jobs=1,
                ) -> dict:

    clf, param_grid = get_clf_hyperparameters(estimator_name)

    print('estimator: {}, params: {}'.format(estimator_name, param_grid))

    if cv is not None:
        grid_cv = GridSearchCV(clf, param_grid=param_grid, scoring=scoring_estimator, cv=cv, return_train_score=True, n_jobs=n_jobs)
    else:
        grid_cv = GridSearchCV(clf, param_grid=param_grid, scoring=scoring_estimator, cv=5, return_train_score=True, n_jobs=n_jobs)

    grid_cv.fit(x_train, y_train)

    print('Best hyperparams: {}, best_score: {}'.format(grid_cv.best_params_, grid_cv.best_score_))

    best_clf = grid_cv.best_estimator_
    best_clf.fit(x_train, y_train)
    y_pred = best_clf.predict(x_test)

    dict_metrics = compute_classification_prestations(y_test, y_pred, np.unique(y_test))

    return dict_metrics


def train_several_partitions(x_features,
                             y_label,
                             estimator,
                             scoring,
                             cv,
                             list_seed_values=None,
                             n_jobs=1,
                             ):
    list_dict_metrics = []

    for seed_value in list_seed_values:
        x_train, x_test, y_train, y_test = train_test_split(x_features,
                                                            y_label,
                                                            stratify=y_label,
                                                            test_size=0.2,
                                                            random_state=seed_value)

        dict_metrics = perform_clf(estimator,
                                   scoring,
                                   x_train,
                                   y_train,
                                   x_test,
                                   y_test,
                                   cv,
                                   n_jobs=n_jobs
                                   )

        list_dict_metrics.append(dict_metrics)

    return list_dict_metrics


X, y = load_breast_cancer(return_X_y=True)

n_jobs = 4
loo = LeaveOneOut()
list_total_metrics = []

for estimator_name in ['tabpfn']:
# for estimator_name in ['rf', 'xgb', 'lgb', 'tabpfn']:
    list_metrics = train_several_partitions(X,
                                            y,
                                            estimator=estimator_name,
                                            cv=None,
                                            scoring='roc_auc',
                                            list_seed_values=[2, 4, 6, 7, 8],
                                            n_jobs=n_jobs
                                            )
    save_metrics(list_metrics)



