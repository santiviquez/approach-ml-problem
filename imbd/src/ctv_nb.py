import pandas as pd

from nltk.tokenize import word_tokenize
from sklearn import naive_bayes
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer


if __name__ == "__main__":
    df = pd.read_csv("../input/imbd.csv")
    
    # map positive to 1 and negative to 0
    df.sentiment = df.sentiment.apply(
        lambda x: 1 if x == "positive" else 0
    )
    
    # create a new column called kfold fill with -1
    df["kfold"] = -1
    
    # randomnize the row
    df = df.sample(frac=1).reset_index(drop=True)
    
    # fetch the labels
    y = df.sentiment.values
    
    # initialize the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f
        
    # go over the created folds
    for fold_ in range(5):
        train_df = df[df["kfold"] != fold_].reset_index(drop=True)
        test_df = df[df["kfold"] == fold_].reset_index(drop=True)
        
        # initialize CountVectorizer with word_tokenize form nltk
        count_vec = CountVectorizer(
            tokenizer=word_tokenize,
            token_pattern=None
        )
        
        # fit count_vec on training data reviews
        count_vec.fit(train_df.review)
        
        # transform training and validation data reviews
        xtrain = count_vec.transform(train_df.review)
        xtest = count_vec.transform(test_df.review)
        
        # initialize logistic regression model
        model = naive_bayes.MultinomialNB()
        
        # fit the model on training data reviews and sentiment
        model.fit(xtrain, train_df.sentiment)
        
        # make predictions on test data
        preds = model.predict(xtest)
        
        # calculate accuracy
        accuracy = metrics.accuracy_score(test_df.sentiment, preds)
        
        print(f"Fold: {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")