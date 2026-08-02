import torch


def main():
    print("Hello from resnet18-ddp!")
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
