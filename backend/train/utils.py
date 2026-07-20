import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader
from datasets import load_dataset
from models.config import device

train_tf = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(0.2, 0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_loaders(batch_size=8):
    """Создает train и val лоудеры из локального датасета"""

    train_ds = datasets.ImageFolder(f"data/train", transform=train_tf)
    val_ds = datasets.ImageFolder(f"data/val", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader

def train_one_epoch(model, loader, criterion, optimizer):
    """Обучение одной эпохи с возвратом loss и accuracy"""

    model.train()
    total_loss = 0
    correct = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

def evaluate(model, loader, criterion):
    """Считает loss и accuracy модели"""

    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

def plot_curves(train_acc, val_acc, train_loss, val_loss):
    """Рисует кривую обучения"""

    plt.figure()
    plt.plot(train_acc, label="train_acc")
    plt.plot(val_acc, label="val_acc")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig(f"artifacts/accuracy.png")

    plt.figure()
    plt.plot(train_loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.legend()
    plt.title("Loss")
    plt.savefig(f"artifacts/loss.png")