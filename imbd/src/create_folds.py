import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":
    # read training data
    df = pd.read_csv("../input/imbd.csv")

    # map postive to 1 and negative to 0
    df.sentiment = df.sentiment.apply(
        lambda x: 1 if x == "positive" else 0
    )

    # create kfold column
    df["kfold"] = -1

    # randomnize the rows
    df = df.sample(frac=1.0).reset_index(drop=True)

    # fetch labels
    y = df.sentiment.values

    # initiate kfold class
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f


df.to_csv("../input/imbd_folds.csv", index=False)