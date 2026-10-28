"""Export-friendly depthwise-separable keyword classifier."""

from torch import nn


class DSCNN(nn.Module):
    """Consume NCHW features (N, 1, 101, 40) and return class logits."""

    def __init__(self, classes=12, channels=64, blocks=3):
        super().__init__()
        if min(classes, channels, blocks) < 1:
            raise ValueError("Model dimensions must be positive")
        layers = [
            nn.Conv2d(1, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        ]
        for _ in range(blocks):
            layers.extend(
                [
                    nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    nn.Conv2d(channels, channels, 1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                ]
            )
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Conv2d(channels, classes, 1)

    def forward(self, inputs):
        return self.classifier(self.pool(self.features(inputs))).flatten(1)
