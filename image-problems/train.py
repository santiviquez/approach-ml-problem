import os

import pandas as pd
import numpy as np

import albumentations
import torch

from sklearn import metrics
from sklearn.model_selection import train_test_split

import dataset
import engine
from model import get_model


if __name__ == "__main__":
    data_path = ""
    device = "cpu" # or cuda

    epochs = 10

    df = pd.read_csv(os.path.join(data_path, "train.csv"))

    # fetch all image ids
    images = df.ImageId.values.tolist()

    # a list with image locations
    images = [
        os.path.join(data_path, "train_png", i + ".png") for i in images
    ]

    # binary targets numpy array
    targets = df.target.values

    # fetch model
    model = get_model(pretrained=True)

    # move model to device
    model.to(device)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True
            )
        ]
    )

    # train split instead of kfold. But kfold is a better option
    train_images, valid_images, train_targets, valid_targets = train_test_split(
        images, targets, stratify=targets, random_state=42
    )

    # fetch the ClassificationDataset class
    train_dataset = dataset.ClassicationDataset(
        image_paths=train_images,
        targets=train_targets,
        resize=(227, 227),
        augmentations=aug
    )

    # torch dataloader creates batches of data
    # from classification dataset utils
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4
    )

    # same for validation dataset
        # fetch the ClassificationDataset class
    valid_dataset = dataset.ClassicationDataset(
        image_paths=valid_images,
        targets=valid_targets,
        resize=(227, 227),
        augmentations=aug
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=16, shuffle=True, num_workers=4
    )

    # adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # train and print auc score for all epochs
    for epoch in range(epochs):
        engine.train(train_loader, model, optimizer, device=device)
        predictions, valid_targets = engine.evaluate(
            valid_loader, model, device=device
        )
        roc_auc = metrics.roc_auc_score(valid_targets, predictions)
        print(f"Epcoh={epoch}, Valid ROC AUC={roc_auc}")