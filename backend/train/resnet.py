import torch
import torch.nn as nn
import copy
from torchvision import models
from .utils import (
    train_one_epoch,
    evaluate,
    plot_curves,
    get_loaders
)
from models.config import RESNET_MODEL_PATH
from models.resnet import load_resnet

def train_resnet(train_loader, val_loader, pretrained, save_plot=False):
    """Дообучает и сохраняет ResNet18 модель, выбирая лучшие веса"""

    model = load_resnet(pretrained)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_accs, val_accs = [], []
    train_losses, val_losses = [], []

    best_model_tr_loss = float("inf")
    best_model_tr_acc = 0
    best_model_val_loss = float("inf")
    best_model_val_acc = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    max_epochs = 100
    min_epochs = 10
    patience = 10
    threshold = 0.01
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_accs.append(tr_acc)
        val_accs.append(val_acc)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        if val_loss < best_model_val_loss - threshold:
            best_model_tr_loss = tr_loss
            best_model_tr_acc = tr_acc
            best_model_val_loss = val_loss
            best_model_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())

            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience and epoch+1 >= min_epochs:
            print(f"Ранняя остановка в эпохе {epoch+1}")
            break

    model.load_state_dict(best_model_weights)

    if save_plot:
        plot_curves(train_accs, val_accs, train_losses, val_losses)

    torch.save(model.state_dict(), RESNET_MODEL_PATH)

    return model, best_model_tr_loss, best_model_tr_acc, best_model_val_loss, best_model_val_acc