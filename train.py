import torch

from configs.config import Config
from utils.device import get_device
from utils.seed import set_seed
from utils.model_utils import count_parameters
from models.mlp import MLP


def main():

    config = Config()

    set_seed(config.seed)

    device = get_device()

    print("=" * 50)
    print("Configuration")
    print("=" * 50)
    print(config)
    print(f"\nRunning on: {device}")

    model = MLP(num_classes=10)

    print(model)
    print(f"\nParameters: {count_parameters(model):,}")

    dummy = torch.randn(8, 3, 32, 32)
    output = model(dummy)

    print(output.shape)
    
if __name__ == "__main__":
    main()