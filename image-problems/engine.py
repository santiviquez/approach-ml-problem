import torch
import torch.nn as nn

from tqdm import tqdm


def train(data_loader, model, optimizer, device):
    """
    This function does training for one epoch

    Args:
        data_loader: pytorch dataloader
        model: pytorch model
        optimizer: optimizer, e.g. adam, sgd, ect
        device: cuda/cpu
    """

    # put he model in train mode
    model.train()

    # go over every batch of data in the data loader
    for data in data_loader:
        inputs = data["image"]
        targets = data["targets"]

        # move input/targets to cuda/cpu device
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)

        # zero grad the optimizer
        optimizer.zero_grad()
        # do the forward step of model
        outputs = model(inputs)
        # calculate loss
        loss = nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))
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

    # init lists to store targets and outputs
    final_targets = []
    final_outputs = []

    # we use no_grad context
    with torch.no_grad():
        for data in data_loader:
            inputs = data["image"]
            targets = data["targets"]
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            
            # do the forward step to generate prediction
            outputs = model(inputs)

            # convert targets and outputs to list
            targets = targets.detach().cpu().numpy().tolist()
            outputs = outputs.detach().cpu().numpy().tolist()

            # extend the original list
            final_targets.extend(targets)
            final_outputs.extend(outputs)

    return final_outputs, final_targets
