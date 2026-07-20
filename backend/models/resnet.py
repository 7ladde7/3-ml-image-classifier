import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from .config import RESNET_MODEL_PATH, device

def load_resnet(pretrained):
    """Загружает ResNet18 с бинарным классификатором"""

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 2) # для бинарной классификации

    model.to(device)

    return model

def load_finetuned_resnet():
    """Загружает сохраненную дообученную ResNet18"""

    model = load_resnet()
    model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device))

    model.to(device)

    return model