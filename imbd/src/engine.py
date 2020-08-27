import torch
import torch.nn as nn


def train(data_loader, model, optimizer, device):
    """
    This function does training for one epoch

    Args:
        data_loader: pytorch dataloader
        model: pytorch model (lstm)
        optimizer: optimizer, e.g. adam, sgd, ect
        device: cuda/cpu
    """

    # put he model in train mode
    model.train()

    # go over every batch of data in the data loader
    for data in data_loader:
        reviews = data["review"]
        targets = data["target"]

        # move input/targets to cuda/cpu device
        reviews = reviews.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        # zero grad the optimizer
        optimizer.zero_grad()
        
        # make predictions from the model
        predictions = model(reviews)

        # calculate loss
        loss = nn.BCEWithLogitsLoss()(predictions, targets.view(-1, 1))
        
        # backward step the loss
        loss.backward()
        
        # step optimizer
        optimizer.step()


def evaluate(data_loader, model, device):
    """
    This function does evaluation for one epoch

    Args:
        data_loader: pytorch dataloader
        model: pytorch model
        optimizer: optimizer, e.g. adam, sgd, ect
        device: cuda/cpu
    """
    # put he model in eval mode
    model.eval()

    # init lists to store targets and predictions
    final_targets = []
    final_predictions = []

    # we use no_grad context
    with torch.no_grad():
        for data in data_loader:
            reviews = data["review"]
            targets = data["target"]
            reviews = reviews.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            
            # do the forward step to generate prediction
            predictions = model(reviews)

            # convert targets and predictions to list
            targets = targets.detach().cpu().numpy().tolist()
            predictions = predictions.detach().cpu().numpy().tolist()

            # extend the original list
            final_targets.extend(targets)
            final_predictions.extend(predictions)

    return final_predictions, final_targets