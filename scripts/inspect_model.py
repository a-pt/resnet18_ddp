import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from utils.model_utils import count_parameters
from models.residual_block import ResidualBlock
from models.resnet18 import ResNet18

def device_config():
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

def print_model(model):

    print(model)
    print("=" * 50)

    for i, child in enumerate(model.children()):
        print(f"\nChild {i}")
        print(child)

    print("=" * 50)

    for name, module in model.named_modules():
        print(name, module)

    print("=" * 50)

    print("\nNamed Parameters:")
    for name, param in model.named_parameters():
        print(name, param.shape)

    print("\nNamed Buffers:")
    for name, buffer in model.named_buffers():
        print(name, buffer.shape)

    print("=" * 50)

    print(f"\nTotal Parameters: {count_parameters(model):,}")
    print("=" * 50)

    print("Parameters per Layer")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name}: {params:,}")

    print("=" * 50)

    print("\nState Dict:")
    for key in model.state_dict():
        print(key)

    print("=" * 50)

def test_resblock_identity():
    block = ResidualBlock(in_channels=64, out_channels=64)
    print(block)
    x = torch.randn(8,64,32,32)
    y = block(x)
    print(y.shape)

def test_resblock_projection():
    block = ResidualBlock(in_channels=64, out_channels=128, stride=2)
    print(block)
    x = torch.randn(8,64,32,32)
    y = block(x)
    print(y.shape)


def hook(module, input, output):
    print(module.__class__.__name__, output.shape)

def test_resnet18():
    model = ResNet18(num_classes=10)

    print_model(model)

    x = torch.randn(4, 3, 32, 32)

    handle = model.stage2.register_forward_hook(hook)
    y = model(x)

    handle.remove()

    print("Input Shape :", x.shape)
    print("Output Shape:", y.shape)


    
device_config()
print("=" * 50)
test_resnet18()
print("=" * 50)

    
