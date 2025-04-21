from torchvision.models import efficientnet_b0
import torch.nn as nn

def get_efficientnet():
    model = efficientnet_b0(pretrained=True)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model