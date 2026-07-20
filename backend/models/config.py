import torch

CNN_MODEL_PATH = "artifacts/cnn_model.pt"
RESNET_MODEL_PATH = "artifacts/resnet_finetuned.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

val_ratio = 0.2