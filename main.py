import torch
import torch.nn as nn
from utils.model_utils import count_parameters
from models.mlp import MLP
from models.simple_cnn import SimpleCNN

def main():
    print("Hello from resnet18-ddp!")
    
def device_config():
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

def print_model(model):

    print(model)
    print(f"\nParameters: {count_parameters(model):,}")

    print("Named Modules:")
    for name, module in model.named_modules():
        print(name, "->", module)

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

    print_model(model)

    dummy = torch.randn(4, 3, 32, 32)
    output = model(dummy)
    print(output.shape)

if __name__ == "__main__":
    main()
    device_config()
    test_cnn()
