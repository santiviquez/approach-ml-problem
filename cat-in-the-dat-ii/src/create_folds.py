import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":
    # read training data
    df = pd.read_csv('../input/train.csv')

    # create a new column called kfod and fill it with -1
    df['kfold'] = -1

    # next step is to randomnize de row of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch labels
    y = df.target.values

    # initiate the kfold class form model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new column kfold
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    
    # save the new csv with the kfold column
    df.to_csv('../input/train_folds.csv', index=False)