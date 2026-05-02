import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


class ModelNoProjector(nn.Module):
    """
    Same ResNet-18 backbone as the original SimCLR model, but without
    the projection head. The NT-Xent loss is computed directly on the
    512-dimensional backbone output instead of the 128-dimensional
    projector output.
    """
    def __init__(self):
        super(ModelNoProjector, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder only — no projector head
        self.f = nn.Sequential(*self.f)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        # Return feature twice — once as representation, once for loss
        # Both are the same 512-dim vector since there is no projector
        return F.normalize(feature, dim=-1), F.normalize(feature, dim=-1)