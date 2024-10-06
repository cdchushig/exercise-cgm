import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, roc_curve, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from loader import load_raw_data_exercise, save_stats_dataset
import scipy.stats as stats

cols = ["Index", "FullSumValues", "FullMedian", "FullVar", "FullEntropy", "FullHyper", "FullHypo", "FullAdverse",
        "DawnSumValues", "DawnMedian", "DawnVar", "DawnEntropy","DawnHyper", "DawnHypo", "DawnAdverse",
        "MorningSumValues", "MorningMedian", "MorningVar","MorningEntropy", "MorningHyper", "MorningHypo", "MorningAdverse",
        "AftSumValues", "AftMedian", "AftVar", "AftEntropy","AftHyper", "AftHypo", "AftAdverse", "NightSumValues",
        "NightMedian", "NightVar", "NightEntropy", "NightHyper", "NightHypo", "NightAdverse", "Cat"]


def compute_stats_from_glucose(dfw1: pd.DataFrame,
                               dfw2: pd.DataFrame,
                               cgm_device: str = 'Fingerprick (mg/dL)'
                               ):

    tabdf = pd.DataFrame(columns=cols)
    elsettt = [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    h = cgm_device

    for e in elsettt:
        # df = pd.DataFrame(df).append(new_row, ignore_index=True)
        # df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        dict_new_row_w1 = {
            "Index": e,
            "FullSumValues": dfw1[cgm_device].loc[(dfw1['patient_id'] == e)].sum(),
            "FullMedian": dfw1[cgm_device].loc[(dfw1['patient_id'] == e)].median(),
            "FullVar": dfw1[cgm_device].loc[(dfw1['patient_id'] == e)].var(),
            "FullEntropy": stats.entropy(dfw1[cgm_device].loc[(dfw1['patient_id'] == e)].to_list()),
            "FullHyper": dfw1[cgm_device].loc[(dfw1['patient_id'] == e) & (dfw1[cgm_device] > 180)].count(),
            "FullHypo": dfw1[cgm_device].loc[(dfw1['patient_id'] == e) & (dfw1[cgm_device] < 70)].count(),
            "FullAdverse": dfw1[cgm_device].loc[(dfw1['patient_id'] == e) & (dfw1[cgm_device] < 70)].count() +
                           dfw1[cgm_device].loc[(dfw1['patient_id'] == e) & (dfw1[cgm_device] > 180)].count(),
            "DawnSumValues": dfw1[cgm_device].loc[(dfw1['patient_id'] == e) & (dfw1["Datum"] < 6)].sum(),
            "DawnMedian": dfw1[cgm_device].loc[(dfw1['patient_id'] == e) & (dfw1["Datum"] < 6)].median(),
            "DawnVar": dfw1[cgm_device].loc[(dfw1['patient_id'] == e) & (dfw1["Datum"] < 6)].var(),
            "DawnEntropy": stats.entropy(
                dfw1[cgm_device].loc[(dfw1['patient_id'] == e) & (dfw1["Datum"] < 6)].to_list()),
            "DawnHyper": dfw1[cgm_device].loc[
                (dfw1['patient_id'] == e) & (dfw1[cgm_device] > 180) & (dfw1["Datum"] < 6)].count(),
            "DawnHypo": dfw1[cgm_device].loc[
                (dfw1['patient_id'] == e) & (dfw1[cgm_device] < 70) & (dfw1["Datum"] < 6)].count(),
            "DawnAdverse": dfw1[cgm_device].loc[(dfw1['patient_id'] == e) & (dfw1[cgm_device] < 70) & (
                    dfw1["Datum"] < 6)].count() + dfw1[cgm_device].loc[
                               (dfw1['patient_id'] == e) & (dfw1[cgm_device] > 180) & (
                                       dfw1["Datum"] < 6)].count(),
            "MorningSumValues": dfw1[cgm_device].loc[
                (dfw1['patient_id'] == e) & (dfw1["Datum"] >= 6) & (dfw1["Datum"] < 12)].sum(),
            "MorningMedian": dfw1[cgm_device].loc[
                (dfw1['patient_id'] == e) & (dfw1["Datum"] >= 6) & (dfw1["Datum"] < 12)].median(),
            "MorningVar": dfw1[cgm_device].loc[
                (dfw1['patient_id'] == e) & (dfw1["Datum"] >= 6) & (dfw1["Datum"] < 12)].var(),
            "MorningEntropy": stats.entropy(dfw1[cgm_device].loc[
                                                (dfw1['patient_id'] == e) & (dfw1["Datum"] >= 6) & (
                                                        dfw1["Datum"] < 12)].to_list()),
            "MorningHyper": dfw1[cgm_device].loc[
                (dfw1['patient_id'] == e) & (dfw1[cgm_device] > 180) & (dfw1["Datum"] >= 6) & (
                        dfw1["Datum"] < 12)].count(),
            "MorningHypo": dfw1[cgm_device].loc[
                (dfw1['patient_id'] == e) & (dfw1[cgm_device] < 70) & (dfw1["Datum"] >= 6) & (
                        dfw1["Datum"] < 12)].count(),
            "MorningAdverse": dfw1[cgm_device].loc[(dfw1['patient_id'] == e) & (dfw1[cgm_device] < 70) & (
                    dfw1["Datum"] >= 6) & (dfw1["Datum"] < 12)].count() + dfw1[cgm_device].loc[
                                  (dfw1['patient_id'] == e) & (dfw1[cgm_device] > 180) & (
                                          dfw1["Datum"] >= 6) & (dfw1["Datum"] < 12)].count(),
            "AftSumValues": dfw1[cgm_device].loc[
                (dfw1['patient_id'] == e) & (dfw1["Datum"] >= 12) & (dfw1["Datum"] < 18)].sum(),
            "AftMedian": dfw1[cgm_device].loc[
                (dfw1['patient_id'] == e) & (dfw1["Datum"] >= 12) & (dfw1["Datum"] < 18)].median(),
            "AftVar": dfw1[cgm_device].loc[
                (dfw1['patient_id'] == e) & (dfw1["Datum"] >= 12) & (dfw1["Datum"] < 18)].var(),
            "AftEntropy": stats.entropy(dfw1[cgm_device].loc[
                                            (dfw1['patient_id'] == e) & (dfw1["Datum"] >= 12) & (
                                                    dfw1["Datum"] < 18)].to_list()),
            "AftHyper": dfw1[cgm_device].loc[
                (dfw1['patient_id'] == e) & (dfw1[cgm_device] > 180) & (dfw1["Datum"] >= 12) & (
                        dfw1["Datum"] < 18)].count(),
            "AftHypo": dfw1[cgm_device].loc[
                (dfw1['patient_id'] == e) & (dfw1[cgm_device] < 70) & (dfw1["Datum"] >= 12) & (
                        dfw1["Datum"] < 18)].count(),
            "AftAdverse": dfw1[cgm_device].loc[
                              (dfw1['patient_id'] == e) & (dfw1[cgm_device] < 70) & (dfw1["Datum"] >= 12) & (
                                      dfw1["Datum"] < 18)].count() + dfw1[cgm_device].loc[
                              (dfw1['patient_id'] == e) & (dfw1[cgm_device] > 180) & (dfw1["Datum"] >= 12) & (
                                      dfw1["Datum"] < 18)].count(),
            "NightSumValues": dfw1[cgm_device].loc[(dfw1['patient_id'] == e) & (dfw1["Datum"] >= 18)].sum(),
            "NightMedian": dfw1[cgm_device].loc[(dfw1['patient_id'] == e) & (dfw1["Datum"] >= 18)].median(),
            "NightVar": dfw1[cgm_device].loc[(dfw1['patient_id'] == e) & (dfw1["Datum"] >= 18)].var(),
            "NightEntropy": stats.entropy(
                dfw1[cgm_device].loc[(dfw1['patient_id'] == e) & (dfw1["Datum"] >= 18)].to_list()),
            "NightHyper": dfw1[cgm_device].loc[
                (dfw1['patient_id'] == e) & (dfw1[cgm_device] > 180) & (dfw1["Datum"] >= 18)].count(),
            "NightHypo": dfw1[cgm_device].loc[
                (dfw1['patient_id'] == e) & (dfw1[cgm_device] < 70) & (dfw1["Datum"] >= 18)].count(),
            "NightAdverse": dfw1[cgm_device].loc[(dfw1['patient_id'] == e) & (dfw1[cgm_device] < 70) & (
                    dfw1["Datum"] >= 18)].count() + dfw1[cgm_device].loc[
                                (dfw1['patient_id'] == e) & (dfw1[cgm_device] > 180) & (
                                        dfw1["Datum"] >= 18)].count(),
            "Cat": 0
        }

        tabdf = pd.concat([tabdf, pd.DataFrame([dict_new_row_w1])], ignore_index=True)

    for e in elsettt:
        dict_new_row_w2 = {
            "Index": e,
            "FullSumValues": dfw2[h].loc[(dfw2['patient_id'] == e)].sum(),
            "FullMedian": dfw2[h].loc[(dfw2['patient_id'] == e)].median(),
            "FullVar": dfw2[h].loc[(dfw2['patient_id'] == e)].var(),
            "FullEntropy": stats.entropy(dfw2[h].loc[(dfw2['patient_id'] == e)].to_list()),
            "FullHyper": dfw2[h].loc[(dfw2['patient_id'] == e) & (dfw2[h] > 180)].count(),
            "FullHypo": dfw2[h].loc[(dfw2['patient_id'] == e) & (dfw2[h] < 70)].count(),
            "FullAdverse": dfw2[h].loc[(dfw2['patient_id'] == e) & (dfw2[h] < 70)].count() +
                           dfw2[h].loc[(dfw2['patient_id'] == e) & (dfw2[h] < 70)].count(),
            "DawnSumValues": dfw2[h].loc[(dfw2['patient_id'] == e) & (dfw2["Datum"] < 6)].sum(),
            "DawnMedian": dfw2[h].loc[(dfw2['patient_id'] == e) & (dfw2["Datum"] < 6)].median(),
            "DawnVar": dfw2[h].loc[(dfw2['patient_id'] == e) & (dfw2["Datum"] < 6)].var(),
            "DawnEntropy": stats.entropy(
                dfw2[h].loc[(dfw2['patient_id'] == e) & (dfw2["Datum"] < 6)].to_list()),
            "DawnHyper": dfw2[h].loc[
                (dfw2['patient_id'] == e) & (dfw2[h] > 180) & (dfw2["Datum"] < 6)].count(),
            "DawnHypo": dfw2[h].loc[
                (dfw2['patient_id'] == e) & (dfw2[h] < 70) & (dfw2["Datum"] < 6)].count(),
            "DawnAdverse": dfw2[h].loc[(dfw2['patient_id'] == e) & (dfw2[h] < 70) & (
                    dfw2["Datum"] < 6)].count() + dfw2[h].loc[
                               (dfw2['patient_id'] == e) & (dfw2[h] > 180) & (
                                       dfw2["Datum"] < 6)].count(),
            "MorningSumValues": dfw2[h].loc[
                (dfw2['patient_id'] == e) & (dfw2["Datum"] >= 6) & (dfw2["Datum"] < 12)].sum(),
            "MorningMedian": dfw2[h].loc[
                (dfw2['patient_id'] == e) & (dfw2["Datum"] >= 6) & (dfw2["Datum"] < 12)].median(),
            "MorningVar": dfw2[h].loc[
                (dfw2['patient_id'] == e) & (dfw2["Datum"] >= 6) & (dfw2["Datum"] < 12)].var(),
            "MorningEntropy": stats.entropy(dfw2[h].loc[
                                                (dfw2['patient_id'] == e) & (dfw2["Datum"] >= 6) & (
                                                        dfw2["Datum"] < 12)].to_list()),
            "MorningHyper": dfw2[h].loc[
                (dfw2['patient_id'] == e) & (dfw2[h] > 180) & (dfw2["Datum"] >= 6) & (
                        dfw2["Datum"] < 12)].count(),
            "MorningHypo": dfw2[h].loc[
                (dfw2['patient_id'] == e) & (dfw2[h] < 70) & (dfw2["Datum"] >= 6) & (
                        dfw2["Datum"] < 12)].count(),
            "MorningAdverse": dfw2[h].loc[(dfw2['patient_id'] == e) & (dfw2[h] < 70) & (
                    dfw2["Datum"] >= 6) & (dfw2["Datum"] < 12)].count() + dfw2[h].loc[
                                  (dfw2['patient_id'] == e) & (dfw2[h] > 180) & (
                                          dfw2["Datum"] >= 6) & (dfw2["Datum"] < 12)].count(),
            "AftSumValues": dfw2[h].loc[
                (dfw2['patient_id'] == e) & (dfw2["Datum"] >= 12) & (dfw2["Datum"] < 18)].sum(),
            "AftMedian": dfw2[h].loc[
                (dfw2['patient_id'] == e) & (dfw2["Datum"] >= 12) & (dfw2["Datum"] < 18)].median(),
            "AftVar": dfw2[h].loc[
                (dfw2['patient_id'] == e) & (dfw2["Datum"] >= 12) & (dfw2["Datum"] < 18)].var(),
            "AftEntropy": stats.entropy(dfw2[h].loc[
                                            (dfw2['patient_id'] == e) & (dfw2["Datum"] >= 12) & (
                                                    dfw2["Datum"] < 18)].to_list()),
            "AftHyper": dfw2[h].loc[
                (dfw2['patient_id'] == e) & (dfw2[h] > 180) & (dfw2["Datum"] >= 12) & (
                        dfw2["Datum"] < 18)].count(),
            "AftHypo": dfw2[h].loc[
                (dfw2['patient_id'] == e) & (dfw2[h] < 70) & (dfw2["Datum"] >= 12) & (
                        dfw2["Datum"] < 18)].count(),
            "AftAdverse": dfw2[h].loc[
                              (dfw2['patient_id'] == e) & (dfw2[h] < 70) & (dfw2["Datum"] >= 12) & (
                                      dfw2["Datum"] < 18)].count() + dfw2[h].loc[
                              (dfw2['patient_id'] == e) & (dfw2[h] > 180) & (dfw2["Datum"] >= 12) & (
                                      dfw2["Datum"] < 18)].count(),
            "NightSumValues": dfw2[h].loc[(dfw2['patient_id'] == e) & (dfw2["Datum"] >= 18)].sum(),
            "NightMedian": dfw2[h].loc[(dfw2['patient_id'] == e) & (dfw2["Datum"] >= 18)].median(),
            "NightVar": dfw2[h].loc[(dfw2['patient_id'] == e) & (dfw2["Datum"] >= 18)].var(),
            "NightEntropy": stats.entropy(
                dfw2[h].loc[(dfw2['patient_id'] == e) & (dfw2["Datum"] >= 18)].to_list()),
            "NightHyper": dfw2[h].loc[
                (dfw2['patient_id'] == e) & (dfw2[h] > 180) & (dfw2["Datum"] >= 18)].count(),
            "NightHypo": dfw2[h].loc[
                (dfw2['patient_id'] == e) & (dfw2[h] < 70) & (dfw2["Datum"] >= 18)].count(),
            "NightAdverse": dfw2[h].loc[(dfw2['patient_id'] == e) & (dfw2[h] < 70) & (
                    dfw2["Datum"] >= 18)].count() + dfw2[h].loc[
                                (dfw2['patient_id'] == e) & (dfw2[h] > 180) & (
                                        dfw2["Datum"] >= 18)].count(),
            "Cat": 1
        }

        tabdf = pd.concat([tabdf, pd.DataFrame([dict_new_row_w2])], ignore_index=True)

    return tabdf


def repeated_evaluation(classifier_name: str, tabdf_later: pd.DataFrame, selected_variables):
    acc_list = []
    f1_list = []
    sensitivity_list = []
    specificity_list = []

    train_indexes = [[22, 21, 2, 10, 1, 3, 6, 11, 24, 9, 19, 8, 5, 17, 25, 14],
                     [6, 4, 21, 24, 10, 18, 9, 1, 14, 23, 3, 13, 17, 11, 25, 8],
                     [10, 25, 19, 24, 13, 8, 6, 18, 21, 11, 14, 17, 9, 5, 23, 2],
                     [17, 18, 9, 19, 3, 12, 14, 23, 20, 24, 5, 13, 11, 4, 2, 15],
                     [2, 24, 13, 10, 19, 5, 8, 4, 25, 3, 18, 12, 20, 21, 1, 22]]
    test_indexes = [[12, 13, 15, 18, 20, 23], [2, 5, 12, 15, 19, 20, 22], [1, 3, 4, 12, 15, 20, 22],
                    [1, 6, 8, 10, 21, 22, 25], [6, 9, 11, 14, 15, 17, 23]]

    selected_variables.append("Patients")

    X = tabdf_later[selected_variables]
    y = tabdf_later[["Cat", "Patients"]]

    for idx, values in enumerate(train_indexes):
        X_train = X[X["Patients"].isin(train_indexes[idx])]
        X_train = X_train[selected_variables]
        X_train.drop('Patients', axis=1, inplace=True)
        X_test = X[X["Patients"].isin(test_indexes[idx])]
        X_test = X_test[selected_variables]
        X_test.drop('Patients', axis=1, inplace=True)
        y_train = y[y["Patients"].isin(train_indexes[idx])]
        y_train = y_train['Cat']
        y_test = y[y["Patients"].isin(test_indexes[idx])]
        y_test = y_test['Cat']

        scaler = StandardScaler()
        x_train_raw = X_train.copy()
        x_test_raw = X_test.copy()
        scaler.fit(X_train)
        X_train_norm = pd.DataFrame(data=scaler.transform(X_train), columns=list(x_train_raw.columns))
        X_test_norm = pd.DataFrame(data=scaler.transform(X_test), columns=list(x_train_raw.columns))

        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        seed_value = 8
        loo = LeaveOneOut()
        loo.get_n_splits(X_train)

        if classifier_name == 'svm':
            MAX_ITERS = 20000
            dict_param_grid_classifier = {'C': np.linspace(0.01, 10), 'gamma': ['auto', 'scale']}
            grid = GridSearchCV(SVC(max_iter=MAX_ITERS, random_state=seed_value, probability=True),
                                dict_param_grid_classifier, cv=loo, scoring='accuracy', return_train_score=True)
            grid.fit(X_train_norm, y_train)
            best_params = grid.best_params_

            modelo = SVC(**best_params, max_iter=MAX_ITERS, random_state=seed_value, probability=True).fit(X_train_norm,
                                                                                                           y_train)
            y_pred = modelo.predict(X_test_norm)

            acc_test = acc_list.append(accuracy_score(y_test, y_pred))
            f1_list.append(f1_score(y_test, y_pred))
            cmknn = confusion_matrix(y_test, y_pred)
            sensitivity_list.append(cmknn[0][0] / (cmknn[0][0] + cmknn[1][0]))
            specificity_list.append(cmknn[1][1] / (cmknn[1][1] + cmknn[0][1]))

            # explainer = shap.KernelExplainer(modelo.predict_proba, X_test_norm)
            # shap_values = explainer.shap_values(X_test_norm)
            # shap_expected = explainer.expected_value
            # plt.figure()
            # shap.summary_plot(shap_values[1], X_test_norm, plot_type="bar", show=False)
            # name_sum = 'fp_shapbar_' + classifier_name + str(idx) + '.pdf'
            # plt.savefig(name_sum, bbox_inches='tight')

        elif classifier_name == 'knn':
            dict_param_grid_classifier = {'n_neighbors': np.arange(1, 12, 1)}
            grid = GridSearchCV(
                KNeighborsClassifier(),
                dict_param_grid_classifier,
                cv=loo, scoring='accuracy',
                return_train_score=True
            )
            grid.fit(X_train_norm, y_train)
            best_params = grid.best_params_

            modelo = KNeighborsClassifier(**best_params).fit(X_train_norm, y_train)
            y_pred = modelo.predict(X_test_norm)
            acc_test = acc_list.append(accuracy_score(y_test, y_pred))
            f1_list.append(f1_score(y_test, y_pred))
            cmknn = confusion_matrix(y_test, y_pred)
            sensitivity_list.append(cmknn[0][0] / (cmknn[0][0] + cmknn[1][0]))
            specificity_list.append(cmknn[1][1] / (cmknn[1][1] + cmknn[0][1]))

            # SHAP
            # explainer = shap.KernelExplainer(modelo.predict_proba, X_test_norm)
            # shap_values = explainer.shap_values(X_test_norm)
            # shap_expected = explainer.expected_value
            # plt.figure()
            # shap.summary_plot(shap_values[1], X_test_norm, plot_type="bar", show=False)
            # name_sum = 'fp_shapbar_' + classifier_name + str(idx) + '.pdf'
            # plt.savefig(name_sum, bbox_inches='tight')

        elif classifier_name == 'lr':
            dict_param_grid_classifier = {'penalty': ['l2', 'elasticnet', None]}
            grid = GridSearchCV(
                LogisticRegression(random_state=seed_value, solver='liblinear'),
                dict_param_grid_classifier,
                cv=loo,
                scoring='accuracy',
                return_train_score=True
            )
            grid.fit(X_train_norm, y_train)
            best_params = grid.best_params_

            modelo = LogisticRegression(**best_params, random_state=seed_value).fit(X_train_norm, y_train)
            y_pred = modelo.predict(X_test_norm)

            acc_test = acc_list.append(accuracy_score(y_test, y_pred))
            f1_list.append(f1_score(y_test, y_pred))
            cmknn = confusion_matrix(y_test, y_pred)
            sensitivity_list.append(cmknn[0][0] / (cmknn[0][0] + cmknn[1][0]))
            specificity_list.append(cmknn[1][1] / (cmknn[1][1] + cmknn[0][1]))

            # SHAP
            # explainer = shap.KernelExplainer(modelo.predict_proba, X_test_norm)
            # shap_values = explainer.shap_values(X_test_norm)
            # shap_expected = explainer.expected_value
            # plt.figure()
            # shap.summary_plot(shap_values[1], X_test_norm, plot_type="bar", show=False)
            # name_sum = 'fp_shapbar_' + classifier_name + str(idx) + '.pdf'
            # plt.savefig(name_sum, bbox_inches='tight')

    print('acc', np.mean(acc_list), np.std(acc_list))
    print('sensit', np.mean(sensitivity_list), np.std(sensitivity_list))
    print('specfic', np.mean(specificity_list), np.std(specificity_list))
    print('f1', np.mean(f1_list), np.std(f1_list))


def split_glucose_data_into_day_intervals(dfw1: pd.DataFrame, dfw2: pd.DataFrame):

    dict_w1 = {
        'morning': dfw1.loc[(dfw1["Datum"] >= 6) & (dfw1["Datum"] < 12)],
        'afternoon': dfw1.loc[(dfw1["Datum"] >= 12) & (dfw1["Datum"] < 18)],
        'evening': dfw1.loc[(dfw1["Datum"] >= 18)],
        'night': dfw1.loc[(dfw1["Datum"] < 6)],
    }

    dict_w2 = {
        'morning': dfw2.loc[(dfw2["Datum"] >= 6) & (dfw2["Datum"] < 12)],
        'afternoon': dfw2.loc[(dfw2["Datum"] >= 12) & (dfw2["Datum"] < 18)],
        'evening': dfw2.loc[(dfw2["Datum"] >= 18)],
        'night': dfw2.loc[(dfw2["Datum"] < 6)]
    }

    return dict_w1, dict_w2


def parse_arguments(parser):
    parser.add_argument('--n_jobs', default=1, type=int)
    parser.add_argument('--device', default='fsl', type=str)
    return parser.parse_args()


cmd_parser = argparse.ArgumentParser(description='exercise clf')
args = parse_arguments(cmd_parser)

dfw1, dfw2 = load_raw_data_exercise()

dict_devices_mapping = {
    'fingerprick': 'Fingerprick (mg/dL)',
    'eversense': 'Eversense (mg/dL)',
    'fsl': 'FSL (mg/dL)'
}

glucose_device = dict_devices_mapping[args.device]
feature = "patient_id"
list_patient_ids = list(set(dfw1[feature].tolist()))

dfw1_dict = {patient_id: dfw1.loc[dfw1['patient_id'] == patient_id] for patient_id in list_patient_ids}
dfw2_dict = {patient_id: dfw2.loc[dfw2['patient_id'] == patient_id] for patient_id in list_patient_ids}

dfw1["Period"] = ['PA'] * dfw1.shape[0]
dfw2["Period"] = ['NDA'] * dfw2.shape[0]
dff = pd.concat([dfw1, dfw2], axis=0)
dfw1['Datum'] = dfw1['Datum'].str[11:13]
dfw2['Datum'] = dfw2['Datum'].str[11:13]
dfw1['Datum'] = dfw1['Datum'].astype(float)
dfw2['Datum'] = dfw2['Datum'].astype(float)

tabdf = compute_stats_from_glucose(dfw1, dfw2, glucose_device)
tabdf = tabdf.replace(np.nan, 0)
tabdf.set_index("Index", inplace=True)
tabdf_full = tabdf.copy()
tabdf["Patients"] = list(tabdf.index)
tabdf_later = tabdf.copy()

save_stats_dataset(tabdf_later, args.device)

