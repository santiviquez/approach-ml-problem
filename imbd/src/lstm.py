import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, embeding_matrix):
        """
        Args:
            embeding_matrix: numpy arrray with vectors for all words
        """
        super(LSTM, self).__init__()

        num_words = embeding_matrix.shape[0]
        embed_dim = embeding_matrix.shape[1]

        # define an input layer
        self.embedding = nn.Embedding(
            num_embeddings=num_words,
            embedding_dim=embed_dim
        )

        # embedding matrix is used as weights of
        # the embedding layer
        self.embedding.weight = nn.Parameter(
            torch.tensor(
                embedding_matrix,
                dtype=torch.float32
            )
        )

        self.embedding.weight.requires_grad = False

        # simple bidirectional LSTM
        # hidden size of 128
        self.lstm = nn.LSTM(
            embed_dim,
            128,
            bidirectional=True,
            batch_first=True
        )

        # output layer
        self.out = nn.Linear(512, 1)


        def forward(self, x):
            x = self.embedding(x)

            # move embedding output to lstm
            x, _ = self.lstm(x)

            # apply mean and max pooling to lstm
            avg_pool = torch.mean(x, 1)
            max_pool = torch.max(x, 1)

            # concatenate mean and max pooling
            out = torch.cat((avg_pool, max_pool), 1)
            
            # pass through the output layer
            out = self.out(out)

            # return linear output
            return out