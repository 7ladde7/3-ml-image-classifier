import os
import shutil
import requests
import torch
from io import BytesIO
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ddgs import DDGS
from PIL import Image
from torchvision import transforms as T
from train.resnet import train_resnet
from train.utils import get_loaders
from models.config import val_ratio, device
from models.resnet import load_finetuned_resnet
from fastapi.staticfiles import StaticFiles

ddgs = DDGS()

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

app.mount("/artifacts", StaticFiles(directory="artifacts"), name="artifacts")

@app.post("/train")
def train(urls: tuple[list[str], list[str]]):
    """Обучение бинарного классификатора на пользовательских изображениях"""

    # почистим данные, которые могли остаться от предыдущих обучений
    for folder in ["data/train/0", "data/val/0", "data/train/1", "data/val/1"]:
        if os.path.isdir(folder):
            shutil.rmtree(folder)

        os.makedirs(folder, exist_ok=True)

    for l in range(2):
        label_urls = urls[l]

        for i in range(len(label_urls)):
            url = label_urls[i]
            response = requests.get(label_urls[i])
            img = Image.open(BytesIO(response.content))
            img = img.convert("RGB")

            img.save(f"data/{"val" if i >= len(label_urls) * (1 - val_ratio) else "train"}/{l}/{hash(url)}.jpg", "JPEG")

    train_loader, val_loader = get_loaders()

    train_resnet(train_loader, val_loader, pretrained=True, save_plot=True)

    return

@app.get("/search")
def search(query: str):
    """Поиск изображений для обучения"""

    results = ddgs.images(
        query=query,
        max_results=100,
        format="list"
    )

    return [i["thumbnail"] for i in results]

@app.get("/predict")
def predict(url: str):
    """Предсказывает класс изображения из URL"""

    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.convert("RGB")
    x = transforms(img).unsqueeze(0)

    model = load_finetuned_resnet()
    model.eval()

    with torch.no_grad():
        x = x.to(device)
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs, dim=1).values.item()

    return {
        "url": url,
        "prediction": pred,
        "confidence": confidence
    }

    