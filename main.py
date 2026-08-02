import torch
import torch.nn as nn
from utils.model_utils import count_parameters
from models.mlp import MLP
from models.simple_cnn import SimpleCNN
from models.residual_block import ResidualBlock

def device_config():
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

def print_model(model):

    print(model)
    print(f"\nParameters: {count_parameters(model):,}")

    print("\nNamed Parameters:")
    for name, param in model.named_parameters():
        print(name, param.shape)

    print("\nState Dict:")
    for key in model.state_dict():
        print(key)

def test_mlp():
    model = MLP(num_classes=10)
    print_model(model)
    dummy = torch.randn(8, 3, 32, 32)
    output = model(dummy)
    print(output.shape)

def test_cnn():
    model = SimpleCNN()
    print(model)
    dummy = torch.randn(4, 3, 32, 32)
    output = model(dummy)
    print(output.shape)

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

def main():
    device_config()
    test_resblock_identity()
    test_resblock_projection()

if __name__ == "__main__":
    main()
    
