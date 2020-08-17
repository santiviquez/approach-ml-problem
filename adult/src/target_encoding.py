import copy
import pandas as pd

from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb


def mean_target_encoding(data):

    # make a copy of the dataframe
    df = copy.deepcopy(data)
    
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

    # a list to store 5 validation datasets
    encoded_dfs = []

    # go over all folds
    for fold in range(5):
        # get training data using folds
        df_train = df[df['kfold'] != fold].reset_index(drop=True)
        # get validation data using folds
        df_valid = df[df['kfold'] == fold].reset_index(drop=True)

        for column in features:
            # create a dic of category:mean target
            mapping_dic = dict(
                df_train.groupby(column)['income'].mean()
            )
            df_valid.loc[:, column + '_enc'] = df_valid[column].map(mapping_dic)

            # append to our list of enconded validation dataframes
            encoded_dfs.append(df_valid)
    enconded_df = pd.concat(encoded_dfs, axis=0)
    return enconded_df


def run(df, fold):
    # get training data using folds
    df_train = df[df['kfold'] != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    # all columns are features except kfold columns
    features = [
        f for f in df.columns if f not in ('kfold', 'income')
    ]

    # get training data
    x_train = df_train[features].values

    # get validation data
    x_valid = df_valid[features].values

    # initialize XGBoost model
    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7
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
    # read data
    df = pd.read_csv('../input/adult_folds.csv')
    
    # create mean target encoded categories
    df = mean_target_encoding(df)
    
    # run training and validation for 5 folds
    for fold_ in range(5):
        run(df, fold_)