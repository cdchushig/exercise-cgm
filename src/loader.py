import pandas as pd
from pathlib import Path

PATH_PROJECT_DIR = Path(__file__).resolve().parents[1]
PATH_PROJECT_DATA = Path.joinpath(PATH_PROJECT_DIR, 'data')


def load_raw_data_exercise():
    dfw1 = pd.read_excel(str(Path.joinpath(PATH_PROJECT_DATA, "Week 1.xlsx")))
    dfw2 = pd.read_excel(str(Path.joinpath(PATH_PROJECT_DATA, "Week 2.xlsx")))

    dfw1['Datum'] = dfw1['Datum'].astype(str)
    dfw2['Datum'] = dfw2['Datum'].astype(str)

    return dfw1, dfw2


def save_stats_dataset(df_stats: pd.DataFrame, device_name: str):
    df_stats.to_csv(str(Path.joinpath(PATH_PROJECT_DATA, "df_stats_{}.csv".format(device_name))), index=False)


def load_preprocessed_dataset(device_name: str):
    df_data = pd.read_csv(str(Path.joinpath(PATH_PROJECT_DATA, "df_stats_{}.csv".format(device_name))))
    return df_data
