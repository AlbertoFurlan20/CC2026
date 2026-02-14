import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from public_timeseries_testing_util import MockApi
from sklearn.utils import check_consistent_length

def smapep1(y_true, y_pred):
    """SMAPE of y+1, a nonnegative float, smaller is better
    
    Parameters: y_true, y_pred: array-like
    
    Returns 100 for 100 % error.
    y_true may have missing values.
    """
    check_consistent_length(y_true, y_pred)
    y_true = np.array(y_true, copy=False).ravel()
    y_pred = np.array(y_pred, copy=False).ravel()
    y_true, y_pred = y_true[np.isfinite(y_true)], y_pred[np.isfinite(y_true)]
    if (y_true < 0).any(): raise ValueError('y_true < 0')
    if (y_pred < 0).any(): raise ValueError('y_pred < 0')
    denominator = (y_true + y_pred) / 2 + 1
    ape = np.abs(y_pred - y_true) / denominator
    return np.average(ape) * 100

def get_predictions(my_train, model):
    my_train = my_train.fillna(0)
    result = pd.DataFrame(columns=['prediction_id', 'rating'])
    final = []
    target = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]

    for u in target:
        X = my_train["visit_month"]
        predict = model[u].predict(X.values.reshape(-1, 1)).tolist()
        complete_result = my_train[["visit_id", 'visit_month']].values.tolist()
        for index in range(len(complete_result)):
            complete_result[index].extend(predict[index])
        temp = pd.DataFrame(complete_result,
                            columns=["visit_id", 'visit_month', u + '_plus_0_months',
                                     u + '_plus_6_months',
                                     u + '_plus_12_months',
                                     u + '_plus_24_months'])
        temp = temp.melt(id_vars=["visit_id", 'visit_month'],
                         value_vars=[u + '_plus_0_months', u + '_plus_6_months',
                                     u + '_plus_12_months', u + '_plus_24_months'],
                         value_name='rating')
        temp['prediction_id'] = temp['visit_id'] + '_' + temp['variable']
        final.append(temp[['prediction_id', 'rating']])
    final = pd.concat(final)
    final = final.drop_duplicates(subset=['prediction_id', 'rating'])
    return final

if __name__ == "__main__":
    target = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]
    data_proteins = pd.read_csv('train_proteins.csv')
    data_clinical = pd.read_csv('train_clinical_data.csv')
    data_peptides = pd.read_csv('train_peptides.csv')
    data_supplemental = pd.read_csv('supplemental_clinical_data.csv')
    merged_data = pd.concat([data_clinical, data_supplemental])

    id_list = merged_data['patient_id'].unique().tolist()
    data_for_train = {}
    for u in target:
        final = []
        for id_ in id_list:
            infor_of_id = merged_data[merged_data['patient_id'] == id_]
            month_per_id = infor_of_id.visit_month.tolist()
            for month in month_per_id:
                check = [month, id_]
                for plus in [0, 6, 12, 24]:
                    if month + plus in month_per_id:
                        month_value = infor_of_id[infor_of_id.visit_month == month + plus][u].values[0]
                        if not np.isnan(month_value):
                            check.append(month_value)
                if len(check) == 6:
                    final.append(check)
        check = pd.DataFrame(final, columns=['month', 'patient_id', u + '+0', u + '+6', u + '+12', u + '+24'])
        data_for_train[u] = check.dropna()

    model = {}
    overall_score = []
    for i, u in enumerate(target):
        X = data_for_train[u][['month']]
        y = data_for_train[u][u + '+0']
        model[u] = LinearRegression().fit(X, y)
        y_pred = model[u].predict(X)
        score = smapep1(y, y_pred)
        overall_score.append(score)