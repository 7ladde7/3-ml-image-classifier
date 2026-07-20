import torch
torch.random.seed()
import random
random.seed(42)
import matplotlib.pyplot as plt
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import recall_score, precision_score, f1_score
from train.cnn import train_cnn
from train.resnet import train_resnet
from models.config import device, val_ratio
from train.utils import get_loaders, train_tf, val_tf

def evaluate(model, loader):
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()

            y_true.append(y)
            y_pred.append(pred)

    print(f"recall={recall_score(y_true, y_pred):.4f}")
    print(f"precision={precision_score(y_true, y_pred):.4f}")
    print(f"f1={f1_score(y_true, y_pred):.4f}")

train_ds_full = datasets.OxfordIIITPet(
    root="./data",
    target_types="binary-category",
    transform=train_tf,
    download=True
)

val_ds_full = datasets.OxfordIIITPet(
    root="./data",
    target_types="binary-category",
    transform=val_tf,
    download=True
)

cat_idx, dog_idx = [], []

for i in range(len(train_ds_full)):
    _, label = train_ds_full[i]

    if label == 0:
        cat_idx.append(i)
    else:
        dog_idx.append(i)

for num_samples in [50, 250, 1000]:
    random.shuffle(cat_idx)
    random.shuffle(dog_idx)

    train_h = int(num_samples * (1 - val_ratio) / 2)
    val_h = int((num_samples - train_h * 2) / 2)
    test_h = int(500 / 2) # работу модели проверяем на большом количестве изображений

    train_indices = cat_idx[:train_h] + dog_idx[:train_h]
    val_indices = cat_idx[train_h:train_h+val_h] + dog_idx[train_h:train_h+val_h]
    test_indices = cat_idx[train_h+val_h:train_h+val_h+test_h] + dog_idx[train_h+val_h:train_h+val_h+test_h]

    train_ds = Subset(train_ds_full, train_indices)
    val_ds = Subset(val_ds_full, val_indices)
    test_ds = Subset(val_ds_full, test_indices)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)
    test_loader = DataLoader(test_ds, batch_size=1)

    print(f"\n\n\nCNN (samples={num_samples}):\n")

    model, tr_loss, tr_acc, val_loss, val_acc = train_cnn(train_loader, val_loader)

    print(f"tr_loss={tr_loss:.4f}\ntr_acc={tr_acc:.4f}\nval_loss={val_loss:.4f}\nval_acc={val_acc:.4f}")

    evaluate(model, test_loader)

    for pretrained in [True, False]:
        print(f"\n\n\nResNet18 (samples={num_samples}, pretrained={pretrained}):\n")

        model, tr_loss, tr_acc, val_loss, val_acc = train_resnet(train_loader, val_loader, pretrained=pretrained)

        print(f"tr_loss={tr_loss:.4f}\ntr_acc={tr_acc:.4f}\nval_loss={val_loss:.4f}\nval_acc={val_acc:.4f}")

        evaluate(model, test_loader)
