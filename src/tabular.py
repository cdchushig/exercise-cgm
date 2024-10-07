import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from tabpfn import TabPFNClassifier
from loader import load_preprocessed_dataset
from evaluator import compute_classification_prestations

PATH_PROJECT_DIR = Path(__file__).resolve().parents[1]
PATH_PROJECT_METRICS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'metrics')
MAX_ITERS = 20000

list_train_indices = [
    [22, 21, 2, 10, 1, 3, 6, 11, 24, 9, 19, 8, 5, 17, 25, 14],
    [6, 4, 21, 24, 10, 18, 9, 1, 14, 23, 3, 13, 17, 11, 25, 8],
    [10, 25, 19, 24, 13, 8, 6, 18, 21, 11, 14, 17, 9, 5, 23, 2],
    [17, 18, 9, 19, 3, 12, 14, 23, 20, 24, 5, 13, 11, 4, 2, 15],
    [2, 24, 13, 10, 19, 5, 8, 4, 25, 3, 18, 12, 20, 21, 1, 22]
]

list_test_indices = [
    [12, 13, 15, 18, 20, 23],
    [2, 5, 12, 15, 19, 20, 22],
    [1, 3, 4, 12, 15, 20, 22],
    [1, 6, 8, 10, 21, 22, 25],
    [6, 9, 11, 14, 15, 17, 23]
]


def save_metrics(list_total_metrics: list):
    csv_pathfile = str(Path.joinpath(PATH_PROJECT_METRICS, 'metrics_classification.csv'))

    if os.path.exists(csv_pathfile):
        df_metrics = pd.read_csv(csv_pathfile)
    else:
        df_metrics = pd.DataFrame()

    for dict_metrics in list_total_metrics:
        df_metrics = pd.concat([df_metrics, pd.DataFrame([dict_metrics])], ignore_index=True)

    df_metrics.to_csv(csv_pathfile, index=False)


def get_clf_hyperparameters(classifier: str, seed_value: int, n_vars: int):
    selected_clf = None
    param_grid = {}

    if classifier == 'lr':
        param_grid = {
            'C': np.linspace(0.01, 10),
        }
        selected_clf = LogisticRegression(max_iter=MAX_ITERS, random_state=seed_value, solver='liblinear')
    elif classifier == 'dt':
        param_grid = {
            'max_depth': range(3, 20, 1),
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 3, 4, 5, 6, 7, 8]
        }
        selected_clf = DecisionTreeClassifier(random_state=seed_value)
    elif classifier == 'knn':
        param_grid = {
            'n_neighbors': range(1, 20, 2)
        }
        selected_clf = KNeighborsClassifier()
    elif classifier == 'svm':
        param_grid = {
            'C': np.linspace(0.01, 10),
            'gamma': [0.01, 0.001, 0.0001, 0.00001],
        }
        selected_clf = SVC(max_iter=MAX_ITERS, random_state=seed_value, probability=True)
    elif classifier == 'xgb':
        param_grid = {
            'max_depth': range(1, 16, 1),
            'learning_rate': [0.1, 0.01, 0.001],
            'subsample': [0.5, 0.7, 1],
            'n_estimators': [10, 20, 30, 40]
        }
        selected_clf = xgb.XGBClassifier()

    elif classifier == 'mlp':
        selected_clf = MLPClassifier(max_iter=3000)
        param_grid = {
            'hidden_layer_sizes': [(n_vars, 4), (n_vars, 2), (n_vars,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive'],
        }
    elif classifier == 'rf':
        selected_clf = RandomForestClassifier()
        param_grid = {
            'n_estimators': [10, 20, 30, 40],
            'max_depth': range(1, 16, 2),
        }
    elif classifier == 'tabpfn':
        selected_clf = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
        param_grid = {
            'N_ensemble_configurations': [16, 32],
            'batch_size_inference': [20, 30]
        }

    return selected_clf, param_grid


def perform_clf(estimator_name: str,
                scoring_estimator,
                x_train: np.array,
                y_train: np.array,
                x_test: np.array,
                y_test: np.array,
                seed_value,
                cv,
                type_features,
                device,
                n_jobs=1,
                ) -> dict:

    n_vars = x_train.shape[1]
    clf, param_grid = get_clf_hyperparameters(estimator_name, seed_value, n_vars)

    print('estimator: {}, params: {}'.format(estimator_name, param_grid))

    grid_cv = GridSearchCV(clf, param_grid=param_grid, scoring=scoring_estimator, cv=cv, return_train_score=True,
                           n_jobs=n_jobs)

    grid_cv.fit(x_train, y_train)

    print('Best hyperparams: {}, best_score: {}'.format(grid_cv.best_params_, grid_cv.best_score_))

    best_clf = grid_cv.best_estimator_
    best_clf.fit(x_train, y_train)
    y_pred = best_clf.predict(x_test)

    dict_metrics = compute_classification_prestations(y_test, y_pred, np.unique(y_test))

    dict_metrics['seed'] = seed_value
    dict_metrics['estimator'] = estimator_name
    dict_metrics['features'] = type_features
    dict_metrics['device'] = device

    return dict_metrics


def train_test_split_subsets(X, y, selected_variables, idx):
    X_train = X[X["Patients"].isin(list_train_indices[idx])]
    X_train = X_train[selected_variables]
    # X_train.drop('Patients', axis=1, inplace=True)
    X_test = X[X["Patients"].isin(list_test_indices[idx])]
    X_test = X_test[selected_variables]
    # X_test.drop('Patients', axis=1, inplace=True)
    y_train = y[y["Patients"].isin(list_train_indices[idx])]
    y_train = y_train['Cat']
    y_test = y[y["Patients"].isin(list_test_indices[idx])]
    y_test = y_test['Cat']

    return X_train, y_train, X_test, y_test


def scale_data(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    x_train_raw = X_train.copy()
    scaler.fit(X_train)
    df_train_norm = pd.DataFrame(data=scaler.transform(X_train), columns=list(x_train_raw.columns))
    df_test_norm = pd.DataFrame(data=scaler.transform(X_test), columns=list(x_train_raw.columns))

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    return df_train_norm, y_train, df_test_norm, y_test


def train_several_partitions(df_data,
                             type_features,
                             device,
                             estimator,
                             scoring,
                             list_seed_values=None,
                             n_jobs=1,
                             ):
    list_dict_metrics = []

    selected_variables = filter_selected_variables(type_features)
    X = df_data
    y = df_data[["Cat", "Patients"]]

    for idx, values in enumerate(list_train_indices):
        x_train, y_train, x_test, y_test = train_test_split_subsets(X, y, selected_variables, idx)
        x_train_norm, y_train, x_test_norm, y_test = scale_data(x_train, y_train, x_test, y_test)

        loo = LeaveOneOut()
        loo.get_n_splits(x_train)

        dict_metrics = perform_clf(estimator,
                                   scoring,
                                   x_train_norm,
                                   y_train,
                                   x_test_norm,
                                   y_test,
                                   idx,
                                   loo,
                                   type_features,
                                   device,
                                   n_jobs=n_jobs
                                   )

        list_dict_metrics.append(dict_metrics)

    return list_dict_metrics


def filter_selected_variables(type_features: str):

    if type_features == 'all':
        selected_features = ["FullAdverse", "DawnMedian", "DawnSumValues", "DawnVar", "NightMedian", "AftVar",
                             "NightHyper", "DawnHypo", "MorningEntropy", "MorningHypo", "NightHypo"]
    elif type_features == 'morning':
        selected_features = ['MorningSumValues', 'MorningMedian', 'MorningVar', 'MorningHypo', 'MorningAdverse']
    elif type_features == 'afternoon':
        selected_features = ['AftMedian', 'AftVar', 'AftEntropy', 'AftHyper', 'AftHypo']
    elif type_features == 'evening':
        selected_features = ['NightMedian', 'NightVar', 'NightEntropy', 'NightHyper', 'NightHypo']
    elif type_features == 'night':
        selected_features = ['DawnMedian', 'DawnVar', 'DawnEntropy', 'DawnHyper', 'DawnHypo']
    elif type_features == 'full':
        selected_features = ['FullSumValues', 'FullMedian', 'FullVar', 'FullEntropy', 'FullHypo', 'FullAdverse']
    else:
        selected_features = ["FullAdverse", "DawnMedian", "DawnSumValues", "DawnVar", "NightMedian", "AftVar",
                             "NightHyper", "DawnHypo", "MorningEntropy", "MorningHypo", "NightHypo"]

    return selected_features


def compute_mean_metrics(list_dict_metrics: list):
    print(list_dict_metrics)
    for metric in ['accuracy', 'precision', 'specificity', 'recall', 'roc_auc', 'f1']:
        v_metrics = np.array(list(map(lambda d: d[metric], list_dict_metrics)))
        print(metric, np.mean(v_metrics), np.std(v_metrics))


def parse_arguments(parser):
    parser.add_argument('--n_jobs', default=1, type=int)
    parser.add_argument('--clf', default='rf', type=str)
    parser.add_argument('--device', default='fsl', type=str)
    parser.add_argument('--features', default='all', type=str)
    return parser.parse_args()


cmd_parser = argparse.ArgumentParser(description='exercise clf')
args = parse_arguments(cmd_parser)

df_data = load_preprocessed_dataset(args.device)

list_total_metrics = []

for estimator_name in ['lr', 'dt', 'knn', 'svm', 'rf', 'xgb', 'mlp', 'tabpfn']:
    list_dict_metrics = train_several_partitions(df_data,
                                                 args.features,
                                                 args.device,
                                                 estimator=estimator_name,
                                                 scoring='accuracy',
                                                 list_seed_values=[2, 4, 6, 7, 8],
                                                 n_jobs=args.n_jobs
                                                 )
    compute_mean_metrics(list_dict_metrics)
    save_metrics(list_dict_metrics)



