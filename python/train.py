import torch
import torch.nn as nn
import torch.nn.utils

from model import *
from dataset import *

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def train(
        model: nn.Module, 
        dataloader, 
        loss_fn, 
        optimizer
) -> tuple[int, float]:
    model.train()

    for step, (features, targets) in enumerate(dataloader):
        features, targets = features.to(torch.float32).to(device), targets.to(torch.float32).to(device)

        preds = model(features)
        loss = loss_fn(preds.squeeze(), targets.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        yield step, loss.item()

@torch.no_grad()
def eval(
        model: nn.Module, 
        dataloader, 
        loss_fn
) -> float:
    model.eval()
    features, targets = next(iter(dataloader))
    features, targets = features.to(torch.float32).to(device), targets.to(torch.float32).to(device)
    preds = model(features)
    loss = loss_fn(preds.squeeze(), targets.squeeze())
    return loss.item()

def solver(
        model: nn.Module,
        train_dataloader,
        eval_dataloader,
        lr: float = 1e-3,
        epochs: int = 5,
        print_every: int = 500
) -> tuple[list[float], list[float]]:

    print(f'Training {type(model).__name__} on {device}...')

    model.to(device)
    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    len_train_dataloader = len(train_dataloader)
    train_loss_history = []
    eval_loss_history = []

    for epoch in range(epochs):
        for step, train_loss in train(
            model, train_dataloader, loss_fn, optimizer
        ):
            eval_loss = eval(model, eval_dataloader, loss_fn)
            # logging
            eval_loss_history.append(eval_loss)
            train_loss_history.append(train_loss)
            # printing
            if step % print_every == 0:
                print(
                    f'Epoch: {epoch+1}/{epochs},',
                    f'Step: {step+1}/{len_train_dataloader},',
                    f'Validation Loss: {train_loss:.5f}'
                )
            # # saving
            # if step % config.save_iterations == 0:
            #     save_model

    return train_loss_history, eval_loss_history