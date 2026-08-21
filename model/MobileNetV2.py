"""MobileNetV2 for CINIC-10 / CIFAR-style 32x32 inputs (torchvision backbone)."""

import torch.nn as nn
import torchvision.models as models


def mobilenetv2(num_classes=10, in_channels=3, **kwargs):
    """Build MobileNetV2 with a custom classifier head."""
    model = models.mobilenet_v2(weights=None)

    if in_channels != 3:
        first = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels,
            first.out_channels,
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            bias=False,
        )

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
