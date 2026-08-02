from configs.config import Config
from utils.device import get_device
from utils.seed import set_seed


def main():

    config = Config()

    set_seed(config.seed)

    device = get_device()

    print("=" * 50)
    print("Configuration")
    print("=" * 50)
    print(config)

    print(f"\nRunning on: {device}")


if __name__ == "__main__":
    main()