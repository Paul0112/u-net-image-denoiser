import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the train and validation setup

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss_sum = 0
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = F.pad(X, (0, 1, 0, 1)) # fix error in the shape due convolution

        # Compute prediction error
        pred = model(X)
        pred = pred[:, :, :63, :63] # return to original shape
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       

        loss_sum += loss.item()

        if batch % 10 == 0:
            current = (batch + 1) * len(X)
            print(f"Train Loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

    # avg loss per epoch
    return loss_sum / num_batches


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    # No gradient
    with torch.no_grad():
        for X, y in dataloader: # evaluate
            X, y = X.to(device), y.to(device)
            
            X = F.pad(X, (0, 1, 0, 1)) # fix error in the shape due convolution
            pred = model(X) # evaluate
            pred = pred[:, :, :63, :63] # return to original shape

            test_loss += loss_fn(pred, y).item()
            
    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")

    return test_loss
