import os
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Full_mamo_deep_learning.efficientnet import get_efficientnet
from Full_mamo_deep_learning.train import train_model

@pytest.fixture
def dummy_dataloader():
    images = torch.randn(16, 1, 224, 224)  # EfficientNet expects 224x224 input
    labels = torch.randint(0, 2, (16,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=8)

def test_efficientnet_training_runs(tmp_path, dummy_dataloader):
    model = get_efficientnet()
    save_path = os.path.join(tmp_path, "efficientnet_test_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    try:
        train_model(
            model,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            device=device,
            save_path=save_path,
            num_epochs=1
        )
        assert os.path.exists(save_path), "Model file was not saved"
    except Exception as e:
        pytest.fail(f"Training failed with error: {e}")