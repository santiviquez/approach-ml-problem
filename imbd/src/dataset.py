import torch


class IMDBDataset:
    def __init__(self, reviews, targets):
        """
        Args:
            reviews: this a numpy array
            targets: a vector, numpy array
        """
        self.reviews = reviews
        self.target = targets
    

    def __len__(self):
        # returns length of thr dataset
        return len(self.reviews)
    

    def __getitem__(self, item):
        # for any given item, wich is an int,
        # return review and targets as torch tensor
        # item is the index of the item in concern
        review = self.reviews[item, :]
        target = self.target[item, :]

        return {
            "review": torch.tensor(review, dtype=torch.long),
            "target": torch.tensor(target, detype=torch.float)
        }