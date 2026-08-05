import torch
import torch.nn as nn

# Define the train and validation setup

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss_sum = 0
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_sum += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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

            pred = model(X) # evaluate
            test_loss += loss_fn(pred, y).item()
            
    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")

    return test_loss
