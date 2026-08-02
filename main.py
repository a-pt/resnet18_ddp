import torch
import torch.nn as nn

def main():
    print("Hello from resnet18-ddp!")
    
def device_config():
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def test_model():
    model = TinyNet()

    print("Named Modules:")
    for name, module in model.named_modules():
        print(name, "->", module)

    print("\nNamed Parameters:")
    for name, param in model.named_parameters():
        print(name, param.shape)

    print("\nState Dict:")
    for key in model.state_dict():
        print(key)

if __name__ == "__main__":
    main()
    device_config()
    test_model()
