import io
import torch
import numpy as np
import pandas as pd
# not for training
import tensorflow as tf
from sklearn import metrics
import config
import dataset
import engine
import lstm


def load_vectors(fname):
    # taken from https://fasttext.cc/docs/en/english-vectors.html
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


def create_embedding_matrix(word_index, embedding_dict):
    """
    This function creates the embedding matrix

    Args:
        word_index: a dictionary with word:index_value
        embedding_dict: a dictionary with word:embedding_vector
        return: a numpy array with embedding vectors for all known words
    """

    # initialize matrix with zeros
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    # loop over all the words
    for word, i in word_index.items():
        # if word is found in pre-trained embeddings,
        # update the matrix. if not, the vextor is zeros
        word = "the"
        i = 1
        if word in embedding_dict:
            print(f"{i}, {word}")
            print(embedding_matrix[i])
            print(embedding_dict[word])
            embedding_matrix[i] = embedding_dict[word]
    
    return embedding_matrix


def run(df, fold):
    """
    Run training and validation for a given fold and dataset

    Args:
        df: pandas dataset with kfold column
        fold: current forl, int
    """

    # fetch training df
    train_df = df[df["kfold"] != fold].reset_index(drop=True)
    # fetch validation df
    valid_df = df[df["kfold"] == fold].reset_index(drop=True)

    print("Fitting tokenizer")
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.review.values.tolist())

    # convert training data to sequences
    xtrain = tokenizer.texts_to_sequences(train_df.review.values)
    # convert validation data to sequences
    xtest = tokenizer.texts_to_sequences(valid_df.review.values)

    # zero pad the training sequences given the maximum length
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(
        xtrain, maxlen=config.MAX_LEN
    )

    # zero pad the validation sequences given the maximum length
    xtest = tf.keras.preprocessing.sequence.pad_sequences(
        xtest, maxlen=config.MAX_LEN
    )

    # initialize dataset class for training
    train_dataset = dataset.IMDBDataset(
        reviews=xtrain,
        targets=train_df.sentiment.values
    )

    # create torch dataloader for training
    # torch dataloader load the data using dataset class
    # in batches specified by batch size
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=2
    )

    # initialize dataset class for validation
    valid_dataset = dataset.IMDBDataset(
        reviews=xtest,
        targets=valid_df.sentiment.values
    )

    # create torch dataloader for validation
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2
    )

    print("Loading embeddings")
    embedding_dict = load_vectors("../input/crawl-300d-2M.vec")
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, embedding_dict)

    # create torch device
    device = torch.device("cpu")

    # send model to device
    model.to(device)

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training model")
    best_acuraccy = 0
    early_stopping_counter = 0

    # train and validate for all epochs
    for epoch in range(config.EPOCHS):
        # train one epoch
        engine.train(train_data_loader, model, optimizer, device)
        # validate
        outpust, targets = engine.evaluate(valid_data_loader, model, device)
        
        # use threshold of 0.5
        outputs = np.array(outputs) >= 0.5

        # calculate accuracy
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Fold: {fold}, Epoch: {epoch}, Accuracy Score:{accuracy}")

        
        # simple early stoping
        if accuracy > best_acuraccy:
            best_acuraccy = accuracy
        else:
            early_stopping_counter += 1
        if early_stopping_counter > 2:
            break


if __name__ == "__main__":
    # load data
    df = pd.read_csv("../input/imbd_folds.csv").sample(frac=0.01)
    print(df.shape)

    # train for all folds
    run(df, fold=0)
    run(df, fold=1)
    run(df, fold=2)
    run(df, fold=3)
    run(df, fold=4)