import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, roc_curve, f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut

from stats import compute_stats_from_glucose


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


PATH_PROJECT_DIR = Path(__file__).resolve().parents[1]
PATH_PROJECT_DATA_RAW = Path.joinpath(PATH_PROJECT_DIR, 'data')

dfw1 = pd.read_excel(str(Path.joinpath(PATH_PROJECT_DATA_RAW, "Week 1.xlsx")))
dfw2 = pd.read_excel(str(Path.joinpath(PATH_PROJECT_DATA_RAW, "Week 2.xlsx")))

dfw1['Datum'] = dfw1['Datum'].astype(str)
dfw2['Datum'] = dfw2['Datum'].astype(str)

dfw1.rename(columns={"Fingerprick  (mg/dL)": "Fingerprick (mg/dL)"}, inplace=True)
feature = "Patiënt ID"
elset = list(set(dfw1[feature].tolist()))

dfw1_dict = {idd: dfw1.loc[dfw1['Patiënt ID'] == idd] for idd in elset}
dfw2_dict = {idd: dfw2.loc[dfw2['Patiënt ID'] == idd] for idd in elset}

dfw1["Period"] = ['PA'] * dfw1.shape[0]
dfw2["Period"] = ['NDA'] * dfw2.shape[0]
dff = pd.concat([dfw1, dfw2], axis=0)

dfw1['Datum'] = dfw1['Datum'].str[11:13]
dfw2['Datum'] = dfw2['Datum'].str[11:13]
dfw1['Datum'] = dfw1['Datum'].astype(float)
dfw2['Datum'] = dfw2['Datum'].astype(float)

tabdf = compute_stats_from_glucose(dfw1, dfw2)
tabdf = tabdf.replace(np.nan, 0)
tabdf.set_index("Index", inplace=True)
tabdf_full = tabdf.copy()
tabdf["Patients"] = list(tabdf.index)
tabdf_later = tabdf.copy()

selectedvar = ["FullAdverse", "DawnMedian", "DawnSumValues", "DawnVar", "NightMedian", "AftVar", "NightHyper",
               "DawnHypo", "MorningEntropy", "MorningHypo", "NightHypo"]

# repeated_evaluation('lr', tabdf_later.copy(), selectedvar)
# repeated_evaluation('svm', tabdf_later.copy(), selectedvar)
repeated_evaluation('knn', tabdf_later.copy(), selectedvar)


# Morning
morning_feat = [i for i in tabdf_full.columns if "Morn" in i]
morning_feat.append("Cat")
morning_feat.append("Patients")
tabdf = tabdf_full
tabdf = tabdf[morning_feat]
tabdf["Patients"] = list(tabdf.index)
# corr_matrix = tabdf.corr()
# corr_pairs = corr_matrix.unstack()
# sorted_pairs = corr_pairs.sort_values(kind="quicksort")
# upper_pairs = sorted_pairs[(sorted_pairs >= 0.85) & (sorted_pairs != 1)]
# upper = pd.DataFrame(upper_pairs)
# upper.columns = ["Corr"]
# upper_res = upper.drop_duplicates(subset=['Corr'], keep='first')
# eliminate_feat = [i[0] for i in upper_res.index]
# tabdf.drop(eliminate_feat, axis = 1, inplace= True)
selectedvariablesmi = ['MorningSumValues', 'MorningMedian', 'MorningVar', 'MorningHypo', 'MorningAdverse']
repeated_evaluation('lr', tabdf_later, selectedvariablesmi)
repeated_evaluation('knn', tabdf_later, selectedvariablesmi)
repeated_evaluation('svm', tabdf_later, selectedvariablesmi)

# Afternoon
aft_feat = [i for i in tabdf_full.columns if "Aft" in i]
aft_feat.append("Cat")
aft_feat.append("Patients")
tabdf = tabdf_full
tabdf = tabdf[aft_feat]
tabdf["Patients"] = list(tabdf.index)
corr_matrix = tabdf.corr()
corr_pairs = corr_matrix.unstack()
sorted_pairs = corr_pairs.sort_values(kind="quicksort")
upper_pairs = sorted_pairs[(sorted_pairs >= 0.85) & (sorted_pairs != 1)]
upper = pd.DataFrame(upper_pairs)
upper.columns = ["Corr"]
upper_res = upper.drop_duplicates(subset=['Corr'], keep='first')

eliminate_feat = [i[1] for i in upper_res.index]
tabdf.drop(eliminate_feat, axis=1, inplace=True)
selectedvariablesmi = ['AftMedian', 'AftVar', 'AftEntropy', 'AftHyper', 'AftHypo']
repeated_evaluation('lr', tabdf_later, selectedvariablesmi)
repeated_evaluation('knn', tabdf_later, selectedvariablesmi)
repeated_evaluation('svm', tabdf_later, selectedvariablesmi)

# Night
night_feat = [i for i in tabdf_full.columns if "Night" in i]
night_feat.append("Cat")
night_feat.append("Patients")
tabdf = tabdf_full
tabdf = tabdf[night_feat]
tabdf["Patients"] = list(tabdf.index)
corr_matrix = tabdf.corr()
corr_pairs = corr_matrix.unstack()
sorted_pairs = corr_pairs.sort_values(kind="quicksort")
upper_pairs = sorted_pairs[(sorted_pairs >= 0.85) & (sorted_pairs != 1)]
upper = pd.DataFrame(upper_pairs)
upper.columns = ["Corr"]
upper_res = upper.drop_duplicates(subset=['Corr'], keep='first')
eliminate_feat = [i[1] for i in upper_res.index]
tabdf.drop(eliminate_feat, axis=1, inplace=True)

selectedvariablesmi = ['NightMedian', 'NightVar', 'NightEntropy', 'NightHyper', 'NightHypo']
repeated_evaluation('lr', tabdf_later, selectedvariablesmi)
repeated_evaluation('knn', tabdf_later, selectedvariablesmi)
repeated_evaluation('svm', tabdf_later, selectedvariablesmi)

# Dawn
dawn_feat = [i for i in tabdf_full.columns if "Dawn" in i]
dawn_feat.append("Cat")
dawn_feat.append("Patients")
tabdf = tabdf_full
tabdf = tabdf[dawn_feat]

tabdf["Patients"] = list(tabdf.index)

corr_matrix = tabdf.corr()
#sns.heatmap(corr_matrix, annot = True)

corr_pairs = corr_matrix.unstack()

sorted_pairs = corr_pairs.sort_values(kind="quicksort")

upper_pairs = sorted_pairs[(sorted_pairs >= 0.85) & (sorted_pairs != 1)]
upper = pd.DataFrame(upper_pairs)
upper.columns = ["Corr"]
upper_res = upper.drop_duplicates(subset=['Corr'], keep='first')
eliminate_feat = [i[1] for i in upper_res.index]
tabdf.drop(eliminate_feat, axis = 1, inplace= True)

selectedvariablesmi =['DawnMedian', 'DawnVar', 'DawnEntropy', 'DawnHyper', 'DawnHypo']
repeated_evaluation('lr', tabdf_later, selectedvariablesmi)
repeated_evaluation('knn', tabdf_later, selectedvariablesmi)
repeated_evaluation('svm', tabdf_later, selectedvariablesmi)

# Full
full_feat = [i for i in tabdf_full.columns if "Full" in i]
full_feat.append("Cat")
full_feat.append("Patients")
tabdf = tabdf_full.copy()
tabdf = tabdf[full_feat[:-1]]
tabdf["Patients"] = list(tabdf.index)
corr_matrix = tabdf.corr()
corr_pairs = corr_matrix.unstack()
sorted_pairs = corr_pairs.sort_values(kind="quicksort")
upper_pairs = sorted_pairs[(sorted_pairs >= 0.85) & (sorted_pairs != 1)]
upper = pd.DataFrame(upper_pairs)
upper.columns = ["Corr"]
upper_res = upper.drop_duplicates(subset=['Corr'], keep='first')
eliminate_feat = [i[0] for i in upper_res.index]
tabdf.drop(eliminate_feat, axis = 1, inplace= True)

selectedvariablesmi = ['FullSumValues', 'FullMedian', 'FullVar', 'FullEntropy', 'FullHypo','FullAdverse']
repeated_evaluation('lr', tabdf_later, selectedvariablesmi)
selectedvariablesmi = ['FullSumValues', 'FullMedian', 'FullVar', 'FullEntropy', 'FullHypo','FullAdverse']
repeated_evaluation('knn', tabdf_later, selectedvariablesmi)
selectedvariablesmi = ['FullSumValues', 'FullMedian', 'FullVar', 'FullEntropy', 'FullHypo','FullAdverse']
repeated_evaluation('svm', tabdf_later, selectedvariablesmi)

random.seed(3)
np.random.seed(3)

selectedvar = ["FullAdverse","DawnMedian", "DawnSumValues", "DawnVar", "NightMedian","AftVar","NightHyper",
               "DawnHypo", "MorningEntropy", "MorningHypo","NightHypo"]
repeated_evaluation('lr', tabdf_later, selectedvar)
repeated_evaluation('knn', tabdf_later, selectedvar)
repeated_evaluation('svm', tabdf_later, selectedvar)