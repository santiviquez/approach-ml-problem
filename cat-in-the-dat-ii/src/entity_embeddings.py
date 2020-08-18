import os
import gc
import joblib
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import  utils


def create_model(data, catcols):
    """
    This function returns a compiled tf.keras model
    for entity embedidings

    Args:
        data: this is a pandas dataframe
        catcols: list of categorical columns names
        return: compiles tf.keras model
    """

    # init list of input for embeddings
    inputs = []

    # init lists of output for embeddings
    outputs = []

    # loop over all catergorical columns
    for c in catcols:
        # find the number of unique values in a column
        num_unique_values = int(data[c].nunique())

        embed_dim = int(min(np.ceil((num_unique_values)/2), 50))

        # simple keras input layer with size 1
        inp = layers.Input(shape=(1,))

        # add embedding layer to raw input
        # embedding size is always 1 more than unique values in input
        out = layers.Embedding(num_unique_values + 1, embed_dim, name=c)(inp)

        out = layers.SpatialDropout1D(0.3)(out)

        # reshape the input to the dimension of the embedding
        # this becomes our output layer for current feature
        out = layers.Reshape(target_shape=(embed_dim,))(out)

        inputs.append(inp)

        outputs.append(out)
    
    # concatenate all output layers
    x = layers.Concatenate()(outputs)

    # add batchnorm layer
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    y = layers.Dense(2, activation="softmax")(x)

    # create final model
    model = Model(inputs=inputs, outputs=y)

    # compile the model
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model


def run(fold):
    # load the full training data with folds
    df = pd.read_csv('../input/train_folds.csv')

    # all columns are features except kfold columns
    features = [
        f for f in df.columns if f not in ('id', 'kfold', 'target')
    ]

    # fill all NaN values with NONE
    for col in features:
        df.loc[:, col] = df[col].fillna("NONE").astype(str)

    # lets label encode all the features
    for col in features:
        # initialize LabelEncoder
        lbl = preprocessing.LabelEncoder()
        # transform all the data
        df.loc[:, col] = lbl.fit_transform(df[col].values)
    
    # get training data using folds
    df_train = df[df['kfold'] != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    # create tf.keras model
    model = create_model(df, features)

    # our features are lists of lists
    xtrain = [
        df_train[features].values[:, k] for k in range(len(features))
    ]
    xvalid = [
        df_valid[features].values[:, k] for k in range(len(features))
    ]

    # fetch target columns
    ytrain = df_train.values
    yvalid = df_valid.values

    # convert target columns to categorical
    ytrain_cat = utils.to_categorical(ytrain)
    yvalid_cat = utils.to_categorical(yvalid)

    # fit the model
    model.fit(xtrain,
                ytrain_cat,
                validation_data=(xvalid, yvalid_cat),
                verbose=1,
                batch_size=1024,
                epochs=3
    )

    # generate validation predictions
    valid_preds = model.predict(xvalid)[:, 1]

    # print roc auc score
    print(metrics.roc_auc_score(yvalid, valid_preds))
    
if __name__ == "__main__":
    run(0)
    run(1)
    run(2)
    run(3)
    run(4)