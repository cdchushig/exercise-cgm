import pandas as pd
import scipy.stats as stats

cols = ["Index", "FullSumValues","FullMedian","FullVar","FullEntropy", "FullHyper", "FullHypo", "FullAdverse",
        "DawnSumValues", "DawnMedian","DawnVar","DawnEntropy","DawnHyper", "DawnHypo", "DawnAdverse",
        "MorningSumValues", "MorningMedian", "MorningVar","MorningEntropy","MorningHyper", "MorningHypo", "MorningAdverse",
        "AftSumValues", "AftMedian","AftVar","AftEntropy","AftHyper", "AftHypo", "AftAdverse", "NightSumValues",
        "NightMedian", "NightVar", "NightEntropy","NightHyper", "NightHypo", "NightAdverse", "Cat"]


def compute_stats_from_glucose(dfw1: pd.DataFrame,
                               dfw2: pd.DataFrame,
                               cgm_device: str = 'Fingerprick (mg/dL)'
                               ):

    tabdf = pd.DataFrame(columns=cols)

    elsettt = [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    h = "Fingerprick (mg/dL)"

    for e in elsettt:
        # df = pd.DataFrame(df).append(new_row, ignore_index=True)
        # df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        dict_new_row_w1 = {
            "Index": e,
            "FullSumValues": dfw1[cgm_device].loc[(dfw1['Patiënt ID'] == e)].sum(),
            "FullMedian": dfw1[cgm_device].loc[(dfw1['Patiënt ID'] == e)].median(),
            "FullVar": dfw1[cgm_device].loc[(dfw1['Patiënt ID'] == e)].var(),
            "FullEntropy": stats.entropy(dfw1[cgm_device].loc[(dfw1['Patiënt ID'] == e)].to_list()),
            "FullHyper": dfw1[cgm_device].loc[(dfw1['Patiënt ID'] == e) & (dfw1[cgm_device] > 180)].count(),
            "FullHypo": dfw1[cgm_device].loc[(dfw1['Patiënt ID'] == e) & (dfw1[cgm_device] < 70)].count(),
            "FullAdverse": dfw1[cgm_device].loc[(dfw1['Patiënt ID'] == e) & (dfw1[cgm_device] < 70)].count() +
                           dfw1[cgm_device].loc[(dfw1['Patiënt ID'] == e) & (dfw1[cgm_device] > 180)].count(),
            "DawnSumValues": dfw1[cgm_device].loc[(dfw1['Patiënt ID'] == e) & (dfw1["Datum"] < 6)].sum(),
            "DawnMedian": dfw1[cgm_device].loc[(dfw1['Patiënt ID'] == e) & (dfw1["Datum"] < 6)].median(),
            "DawnVar": dfw1[cgm_device].loc[(dfw1['Patiënt ID'] == e) & (dfw1["Datum"] < 6)].var(),
            "DawnEntropy": stats.entropy(
                dfw1[cgm_device].loc[(dfw1['Patiënt ID'] == e) & (dfw1["Datum"] < 6)].to_list()),
            "DawnHyper": dfw1[cgm_device].loc[
                (dfw1['Patiënt ID'] == e) & (dfw1[cgm_device] > 180) & (dfw1["Datum"] < 6)].count(),
            "DawnHypo": dfw1[cgm_device].loc[
                (dfw1['Patiënt ID'] == e) & (dfw1[cgm_device] < 70) & (dfw1["Datum"] < 6)].count(),
            "DawnAdverse": dfw1[cgm_device].loc[(dfw1['Patiënt ID'] == e) & (dfw1[cgm_device] < 70) & (
                    dfw1["Datum"] < 6)].count() + dfw1[cgm_device].loc[
                               (dfw1['Patiënt ID'] == e) & (dfw1[cgm_device] > 180) & (
                                       dfw1["Datum"] < 6)].count(),
            "MorningSumValues": dfw1[cgm_device].loc[
                (dfw1['Patiënt ID'] == e) & (dfw1["Datum"] >= 6) & (dfw1["Datum"] < 12)].sum(),
            "MorningMedian": dfw1[cgm_device].loc[
                (dfw1['Patiënt ID'] == e) & (dfw1["Datum"] >= 6) & (dfw1["Datum"] < 12)].median(),
            "MorningVar": dfw1[cgm_device].loc[
                (dfw1['Patiënt ID'] == e) & (dfw1["Datum"] >= 6) & (dfw1["Datum"] < 12)].var(),
            "MorningEntropy": stats.entropy(dfw1[cgm_device].loc[
                                                (dfw1['Patiënt ID'] == e) & (dfw1["Datum"] >= 6) & (
                                                        dfw1["Datum"] < 12)].to_list()),
            "MorningHyper": dfw1[cgm_device].loc[
                (dfw1['Patiënt ID'] == e) & (dfw1[cgm_device] > 180) & (dfw1["Datum"] >= 6) & (
                        dfw1["Datum"] < 12)].count(),
            "MorningHypo": dfw1[cgm_device].loc[
                (dfw1['Patiënt ID'] == e) & (dfw1[cgm_device] < 70) & (dfw1["Datum"] >= 6) & (
                        dfw1["Datum"] < 12)].count(),
            "MorningAdverse": dfw1[cgm_device].loc[(dfw1['Patiënt ID'] == e) & (dfw1[cgm_device] < 70) & (
                    dfw1["Datum"] >= 6) & (dfw1["Datum"] < 12)].count() + dfw1[cgm_device].loc[
                                  (dfw1['Patiënt ID'] == e) & (dfw1[cgm_device] > 180) & (
                                          dfw1["Datum"] >= 6) & (dfw1["Datum"] < 12)].count(),
            "AftSumValues": dfw1[cgm_device].loc[
                (dfw1['Patiënt ID'] == e) & (dfw1["Datum"] >= 12) & (dfw1["Datum"] < 18)].sum(),
            "AftMedian": dfw1[cgm_device].loc[
                (dfw1['Patiënt ID'] == e) & (dfw1["Datum"] >= 12) & (dfw1["Datum"] < 18)].median(),
            "AftVar": dfw1[cgm_device].loc[
                (dfw1['Patiënt ID'] == e) & (dfw1["Datum"] >= 12) & (dfw1["Datum"] < 18)].var(),
            "AftEntropy": stats.entropy(dfw1[cgm_device].loc[
                                            (dfw1['Patiënt ID'] == e) & (dfw1["Datum"] >= 12) & (
                                                    dfw1["Datum"] < 18)].to_list()),
            "AftHyper": dfw1[cgm_device].loc[
                (dfw1['Patiënt ID'] == e) & (dfw1[cgm_device] > 180) & (dfw1["Datum"] >= 12) & (
                        dfw1["Datum"] < 18)].count(),
            "AftHypo": dfw1[cgm_device].loc[
                (dfw1['Patiënt ID'] == e) & (dfw1[cgm_device] < 70) & (dfw1["Datum"] >= 12) & (
                        dfw1["Datum"] < 18)].count(),
            "AftAdverse": dfw1[cgm_device].loc[
                              (dfw1['Patiënt ID'] == e) & (dfw1[cgm_device] < 70) & (dfw1["Datum"] >= 12) & (
                                      dfw1["Datum"] < 18)].count() + dfw1[cgm_device].loc[
                              (dfw1['Patiënt ID'] == e) & (dfw1[cgm_device] > 180) & (dfw1["Datum"] >= 12) & (
                                      dfw1["Datum"] < 18)].count(),
            "NightSumValues": dfw1[cgm_device].loc[(dfw1['Patiënt ID'] == e) & (dfw1["Datum"] >= 18)].sum(),
            "NightMedian": dfw1[cgm_device].loc[(dfw1['Patiënt ID'] == e) & (dfw1["Datum"] >= 18)].median(),
            "NightVar": dfw1[cgm_device].loc[(dfw1['Patiënt ID'] == e) & (dfw1["Datum"] >= 18)].var(),
            "NightEntropy": stats.entropy(
                dfw1[cgm_device].loc[(dfw1['Patiënt ID'] == e) & (dfw1["Datum"] >= 18)].to_list()),
            "NightHyper": dfw1[cgm_device].loc[
                (dfw1['Patiënt ID'] == e) & (dfw1[cgm_device] > 180) & (dfw1["Datum"] >= 18)].count(),
            "NightHypo": dfw1[cgm_device].loc[
                (dfw1['Patiënt ID'] == e) & (dfw1[cgm_device] < 70) & (dfw1["Datum"] >= 18)].count(),
            "NightAdverse": dfw1[cgm_device].loc[(dfw1['Patiënt ID'] == e) & (dfw1[cgm_device] < 70) & (
                    dfw1["Datum"] >= 18)].count() + dfw1[cgm_device].loc[
                                (dfw1['Patiënt ID'] == e) & (dfw1[cgm_device] > 180) & (
                                        dfw1["Datum"] >= 18)].count(),
            "Cat": 0
        }

        tabdf = pd.concat([tabdf, pd.DataFrame([dict_new_row_w1])], ignore_index=True)

    for e in elsettt:
        dict_new_row_w2 = {
            "Index": e,
            "FullSumValues": dfw2[h].loc[(dfw2['Patiënt ID'] == e)].sum(),
            "FullMedian": dfw2[h].loc[(dfw2['Patiënt ID'] == e)].median(),
            "FullVar": dfw2[h].loc[(dfw2['Patiënt ID'] == e)].var(),
            "FullEntropy": stats.entropy(dfw2[h].loc[(dfw2['Patiënt ID'] == e)].to_list()),
            "FullHyper": dfw2[h].loc[(dfw2['Patiënt ID'] == e) & (dfw2[h] > 180)].count(),
            "FullHypo": dfw2[h].loc[(dfw2['Patiënt ID'] == e) & (dfw2[h] < 70)].count(),
            "FullAdverse": dfw2[h].loc[(dfw2['Patiënt ID'] == e) & (dfw2[h] < 70)].count() +
                           dfw2[h].loc[(dfw2['Patiënt ID'] == e) & (dfw2[h] < 70)].count(),
            "DawnSumValues": dfw2[h].loc[(dfw2['Patiënt ID'] == e) & (dfw2["Datum"] < 6)].sum(),
            "DawnMedian": dfw2[h].loc[(dfw2['Patiënt ID'] == e) & (dfw2["Datum"] < 6)].median(),
            "DawnVar": dfw2[h].loc[(dfw2['Patiënt ID'] == e) & (dfw2["Datum"] < 6)].var(),
            "DawnEntropy": stats.entropy(
                dfw2[h].loc[(dfw2['Patiënt ID'] == e) & (dfw2["Datum"] < 6)].to_list()),
            "DawnHyper": dfw2[h].loc[
                (dfw2['Patiënt ID'] == e) & (dfw2[h] > 180) & (dfw2["Datum"] < 6)].count(),
            "DawnHypo": dfw2[h].loc[
                (dfw2['Patiënt ID'] == e) & (dfw2[h] < 70) & (dfw2["Datum"] < 6)].count(),
            "DawnAdverse": dfw2[h].loc[(dfw2['Patiënt ID'] == e) & (dfw2[h] < 70) & (
                    dfw2["Datum"] < 6)].count() + dfw2[h].loc[
                               (dfw2['Patiënt ID'] == e) & (dfw2[h] > 180) & (
                                       dfw2["Datum"] < 6)].count(),
            "MorningSumValues": dfw2[h].loc[
                (dfw2['Patiënt ID'] == e) & (dfw2["Datum"] >= 6) & (dfw2["Datum"] < 12)].sum(),
            "MorningMedian": dfw2[h].loc[
                (dfw2['Patiënt ID'] == e) & (dfw2["Datum"] >= 6) & (dfw2["Datum"] < 12)].median(),
            "MorningVar": dfw2[h].loc[
                (dfw2['Patiënt ID'] == e) & (dfw2["Datum"] >= 6) & (dfw2["Datum"] < 12)].var(),
            "MorningEntropy": stats.entropy(dfw2[h].loc[
                                                (dfw2['Patiënt ID'] == e) & (dfw2["Datum"] >= 6) & (
                                                        dfw2["Datum"] < 12)].to_list()),
            "MorningHyper": dfw2[h].loc[
                (dfw2['Patiënt ID'] == e) & (dfw2[h] > 180) & (dfw2["Datum"] >= 6) & (
                        dfw2["Datum"] < 12)].count(),
            "MorningHypo": dfw2[h].loc[
                (dfw2['Patiënt ID'] == e) & (dfw2[h] < 70) & (dfw2["Datum"] >= 6) & (
                        dfw2["Datum"] < 12)].count(),
            "MorningAdverse": dfw2[h].loc[(dfw2['Patiënt ID'] == e) & (dfw2[h] < 70) & (
                    dfw2["Datum"] >= 6) & (dfw2["Datum"] < 12)].count() + dfw2[h].loc[
                                  (dfw2['Patiënt ID'] == e) & (dfw2[h] > 180) & (
                                          dfw2["Datum"] >= 6) & (dfw2["Datum"] < 12)].count(),
            "AftSumValues": dfw2[h].loc[
                (dfw2['Patiënt ID'] == e) & (dfw2["Datum"] >= 12) & (dfw2["Datum"] < 18)].sum(),
            "AftMedian": dfw2[h].loc[
                (dfw2['Patiënt ID'] == e) & (dfw2["Datum"] >= 12) & (dfw2["Datum"] < 18)].median(),
            "AftVar": dfw2[h].loc[
                (dfw2['Patiënt ID'] == e) & (dfw2["Datum"] >= 12) & (dfw2["Datum"] < 18)].var(),
            "AftEntropy": stats.entropy(dfw2[h].loc[
                                            (dfw2['Patiënt ID'] == e) & (dfw2["Datum"] >= 12) & (
                                                    dfw2["Datum"] < 18)].to_list()),
            "AftHyper": dfw2[h].loc[
                (dfw2['Patiënt ID'] == e) & (dfw2[h] > 180) & (dfw2["Datum"] >= 12) & (
                        dfw2["Datum"] < 18)].count(),
            "AftHypo": dfw2[h].loc[
                (dfw2['Patiënt ID'] == e) & (dfw2[h] < 70) & (dfw2["Datum"] >= 12) & (
                        dfw2["Datum"] < 18)].count(),
            "AftAdverse": dfw2[h].loc[
                              (dfw2['Patiënt ID'] == e) & (dfw2[h] < 70) & (dfw2["Datum"] >= 12) & (
                                      dfw2["Datum"] < 18)].count() + dfw2[h].loc[
                              (dfw2['Patiënt ID'] == e) & (dfw2[h] > 180) & (dfw2["Datum"] >= 12) & (
                                      dfw2["Datum"] < 18)].count(),
            "NightSumValues": dfw2[h].loc[(dfw2['Patiënt ID'] == e) & (dfw2["Datum"] >= 18)].sum(),
            "NightMedian": dfw2[h].loc[(dfw2['Patiënt ID'] == e) & (dfw2["Datum"] >= 18)].median(),
            "NightVar": dfw2[h].loc[(dfw2['Patiënt ID'] == e) & (dfw2["Datum"] >= 18)].var(),
            "NightEntropy": stats.entropy(
                dfw2[h].loc[(dfw2['Patiënt ID'] == e) & (dfw2["Datum"] >= 18)].to_list()),
            "NightHyper": dfw2[h].loc[
                (dfw2['Patiënt ID'] == e) & (dfw2[h] > 180) & (dfw2["Datum"] >= 18)].count(),
            "NightHypo": dfw2[h].loc[
                (dfw2['Patiënt ID'] == e) & (dfw2[h] < 70) & (dfw2["Datum"] >= 18)].count(),
            "NightAdverse": dfw2[h].loc[(dfw2['Patiënt ID'] == e) & (dfw2[h] < 70) & (
                    dfw2["Datum"] >= 18)].count() + dfw2[h].loc[
                                (dfw2['Patiënt ID'] == e) & (dfw2[h] > 180) & (
                                        dfw2["Datum"] >= 18)].count(),
            "Cat": 1
        }

        tabdf = pd.concat([tabdf, pd.DataFrame([dict_new_row_w2])], ignore_index=True)

    return tabdf


def get_hyperparasms():
    lr_param_grid = {"penalty": ["l1", "l2", "none"],
                     "C": [0.01, 0.001, 0.1, 1, 10]}

    svm_param_grid = {"kernel": ["linear", "poly", "rbf", "sigmoid"],
                      "C": [0.01, 0.001, 0.1, 1, 10],
                      "gamma": [0.01, 0.001, 0.1, 1, 10]}

    dt_param_grid = {"criterion": ["gini", "entropy"],
                     "max_depth": [2, 3, 4, 5, 6, 7, 8],
                     "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                     "min_samples_leaf": [2, 3, 4, 5, 6, 7, 8, 9, 10]}

    rf_param_grid = {"n_estimators": np.arange(1, 25, 5),
                     "criterion": ["gini", "entropy"],
                     "max_depth": [2, 3, 4, 5, 6],
                     "min_samples_split": [2, 3, 4, 5, 6],
                     "min_samples_leaf": [2, 3, 4, 5, 6]}

