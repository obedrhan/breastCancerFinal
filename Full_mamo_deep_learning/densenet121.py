from torchvision.models import densenet121
import torch.nn as nn

def get_densenet121():
    model = densenet121(pretrained=True)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model