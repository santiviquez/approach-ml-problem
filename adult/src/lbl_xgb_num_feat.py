import itertools
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn import preprocessing


def feature_engineering(df, cat_cols):
    """
    This function is used for feature engineering
    Args:
        df: dataframe with train/test data
        cat_cols: list of categorical columns
        return: dataframe with new features
    """
    # this will create all 2-combinations of values
    # in the list
    # for example:
    # list(itertools.combinations([1,2,3], 2)) will return
    # [(1, 2), (1, 3), (2, 3)]
    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df.loc[:, c1 + "_" + c2] = df[c1].astype(str) + "_" + df[c2].astype(str)
    return df


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

    # map targets to 0s and 1s
    target_mappings = {
        "<=50K": 0,
        ">50K": 1
    }
    df.loc[:, "income"] = df.income.map(target_mappings)

    # list of categorical variables for feature engineering
    cat_cols = [
        c for c in df.columns if c not in num_cols
        and c not in ("kfold", "income")
    ]

    # add more features
    df = feature_engineering(df, cat_cols)

    # all columns are features except kfold columns
    features = [
        f for f in df.columns if f not in ('kfold', 'income')
    ]

    # fill all NaN values with NONE
    for col in features:
        df.loc[:, col] = df[col].fillna("NONE").astype(str)

    # lets label encode all the features
    for col in features:
        if col not in num_cols:
            # initialize LabelEncoder
            lbl = preprocessing.LabelEncoder()
            # fit LabelEncoder with all the data
            lbl.fit(df[col])
            # transform all the data
            df.loc[:, col] = lbl.transform(df[col])

    # get training data using folds
    df_train = df[df['kfold'] != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    # get training data
    x_train = df_train[features].values

    # get validation data
    x_valid = df_valid[features].values

    # initialize XGBoost model
    model = xgb.XGBClassifier(
        n_jobs=-1
    )

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