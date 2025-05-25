import os
import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import DenseNet constructor and training function
from Full_mamo_deep_learning.densenet121 import get_densenet121
from Full_mamo_deep_learning.train import train_model


@pytest.fixture
def dummy_dataloader():
    # Create dummy data: 10 grayscale images (1x512x512) and 10 labels (0 or 1)
    x = torch.rand(10, 1, 512, 512)
    y = torch.randint(0, 2, (10,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2)
    return loader


def test_densenet_training_runs(tmp_path, dummy_dataloader):
    model = get_densenet121()
    save_path = os.path.join(tmp_path, "densenet_test_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    try:
        train_model(
            model,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            device=device,
            save_path=save_path,
            num_epochs=1  # Keep it fast for testing
        )
        assert os.path.exists(save_path), "Model file was not saved"
    except Exception as e:
        pytest.fail(f"Training failed with error: {e}")