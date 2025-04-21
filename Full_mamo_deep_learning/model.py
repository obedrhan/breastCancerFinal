import torch
import torchvision.transforms as transforms
from torchvision.models import densenet121, efficientnet_b0
from transformers import ViTForImageClassification
from common_utils import MammogramDataset  # your dataset class
import pandas as pd
import os
from torch.utils.data import DataLoader

def convert_grayscale_to_rgb(img):
    return img.convert("RGB")

def get_transforms(model_type):
    if model_type in ["densenet", "efficientnet"]:
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    elif model_type == "vit":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(convert_grayscale_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
def get_model_and_loader(model_type, base_dir, batch_size=16):
    csv_path = os.path.join(base_dir, "data/full_mammogram_paths.csv")
    df = pd.read_csv(csv_path)

    df['pathology'] = df['pathology'].replace('BENIGN_WITHOUT_CALLBACK', 'BENIGN')
    df['label'] = df['pathology'].map({'BENIGN': 0, 'MALIGNANT': 1})

    train_df = df[df['full_path'].str.contains("Training")].reset_index(drop=True)
    test_df = df[df['full_path'].str.contains("Test")].reset_index(drop=True)

    transform = get_transforms(model_type)

    train_dataset = MammogramDataset(train_df, transform=transform)
    test_dataset = MammogramDataset(test_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    if model_type == "densenet":
        model = densenet121(pretrained=True)
        model.features.conv0 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    elif model_type == "efficientnet":
        model = efficientnet_b0(pretrained=True)
        model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    elif model_type == "vit":
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=2,
            ignore_mismatched_sizes=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, train_loader, test_loader