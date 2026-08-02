from models.residual_block import ResidualBlock
import torch
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(64, 64, blocks=2, stride=1)
        self.stage2 = self._make_stage(64, 128, blocks=2, stride=2)
        self.stage3 = self._make_stage(128, 256, blocks=2, stride=2)
        self.stage4 = self._make_stage(256, 512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_stage(self, in_channels, out_channels, blocks, stride):

        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)

        x = torch.flatten(x,1)
        logits = self.fc(x)

        return logits