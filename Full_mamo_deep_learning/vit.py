from torchvision.models import vit_b_16
import torch.nn as nn

def get_vit():
    model = vit_b_16(pretrained=True)
    model.conv_proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    return model