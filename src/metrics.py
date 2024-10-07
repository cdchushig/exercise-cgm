import numpy as np
import pandas as pd
from pathlib import Path
import argparse


PATH_PROJECT_DIR = Path(__file__).resolve().parents[1]
PATH_PROJECT_METRICS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'metrics')
MAX_ITERS = 20000


def load_metrics_file():
    csv_pathfile = str(Path.joinpath(PATH_PROJECT_METRICS, 'metrics_classification.csv'))
    df_metrics = pd.read_csv(csv_pathfile)
    return df_metrics


def filter_data(df_data: pd.DataFrame, device: str, features: str):
    df_data_filtered = df_data.loc[(df_data["device"] == device) & (df_data["features"] == features)]
    return df_data_filtered


def compute_average_metrics(df: pd.DataFrame, device: str, type_features: str):
    list_estimators = np.unique(df['estimator'].values)
    list_metrics = ['accuracy', 'precision', 'specificity', 'recall', 'roc_auc', 'f1']
    for estimator in list_estimators:
        for metric in list_metrics:
            dfx = df.loc[(df["estimator"] == estimator)][metric]
            v_metric_values = dfx.values
            mean_metric = np.mean(v_metric_values)
            std_metric = np.std(v_metric_values)
            print(device, type_features, estimator, metric, mean_metric, std_metric)


def parse_arguments(parser):
    parser.add_argument('--device', default='fsl', type=str)
    parser.add_argument('--features', default='all', type=str)
    return parser.parse_args()


cmd_parser = argparse.ArgumentParser(description='compute metrics')
args = parse_arguments(cmd_parser)

df_metrics = load_metrics_file()
df_metrics_filtered = filter_data(df_metrics, args.device, args.features)

print(df_metrics_filtered)

print(compute_average_metrics(df_metrics_filtered, args.device, args.features))