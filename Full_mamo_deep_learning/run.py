from train import train_model
from model import get_model_and_loader  # adjust if your file names differ
import torch
import os

if __name__ == "__main__":
    BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose model name: "densenet", "efficientnet", or "vit"
    model_name = "vit"
    model, train_loader, test_loader = get_model_and_loader(model_name, BASE_DIR, batch_size=16)

    model_path = os.path.join(BASE_DIR, f"models/full_mamo_deep_learning/{model_name}_full_mammo.pth")
    train_model(model, train_loader, test_loader, device, model_path)