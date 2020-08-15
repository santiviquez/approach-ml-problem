import pandas as pd

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing


def run(fold):
    # load full training data with folds
    df = pd.read_csv('../input/adult_folds.csv')

    # list of numerical columns
    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]

    # drop numerical columns
    df = df.drop(num_cols, axis=1)

    # map targets to 0s and 1s
    target_mappings = {
        "<=50K": 0,
        ">50K": 1
    }
    df.loc[:, "income"] = df.income.map(target_mappings)

    # all columns are features except kfold columns
    features = [
        f for f in df.columns if f not in ('kfold', 'income')
    ]

    # fill all NaN values with NONE
    for col in features:
        df.loc[:, col] = df[col].fillna("NONE").astype(str)

    # get training data using folds
    df_train = df[df['kfold'] != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    # initialize OneHotEncoder from sklearn
    ohe = preprocessing.OneHotEncoder()

    # fit ohe on training and validation features
    full_data = pd.concat(
        [df_train[features], df_valid[features]],
        axis=0
    )
    ohe.fit(full_data[features])

    # transform training data
    x_train = ohe.transform(df_train[features])

    # transform validation data
    x_valid = ohe.transform(df_valid[features])

    # initialize regression model
    model = linear_model.LogisticRegression()

    # fit model on training data
    model.fit(x_train, df_train.income.values)

    # predict on validation data
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    # print auc
    print(f'Fold = {fold}, AUC = {auc}')


if __name__ == '__main__':
    for fold_ in range(5):
        run(fold_)

