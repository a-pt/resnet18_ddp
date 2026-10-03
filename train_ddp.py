import time
import os

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, record_function, ProfilerActivity

from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from utils.checkpoint import save_checkpoint
from configs.config import Config
from utils.seed import set_seed
from datasets.cifar10 import get_cifar10_datasets
from models.resnet18 import ResNet18

from engine_ddp import train_one_epoch_ddp, validate_ddp

def profile_training(model, train_loader, loss_fn, optimizer, scaler, use_amp, device):
    
    model.train()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./runs/resnet18_profiler"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:

        for step, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with record_function("forward"):
                with torch.autocast(
                    device_type=device.type,
                    enabled=use_amp,
                ):
                    logits = model(images)
                    loss = loss_fn(logits, labels)

            with record_function("backward"):
                scaler.scale(loss).backward()

            with record_function("optimizer_step"):
                scaler.step(optimizer)
                scaler.update()

            prof.step()

            if step >= 4:
                break
    
    print("\n" + "=" * 50)
    print("Profiler Results")
    print("=" * 50)
    print(prof.key_averages().table(sort_by="self_cuda_time_total",row_limit=20))

def setup_distributed():
    
    dist.init_process_group(backend="nccl",init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    return rank, world_size, local_rank


def main():
    rank, world_size, local_rank = setup_distributed()

    config = Config()
    set_seed(config.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NCCL DDP.")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    print(
        f"Rank {rank}/{world_size} | "
        f"Local Rank {local_rank} | "
        f"Device {device}"
    )

    # cuDNN benchmark
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False

    # AMP configuration
    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler("cuda",enabled=use_amp)
    
    if rank == 0:
        print("=" * 50)
        print("Configuration")
        print("=" * 50)
        print(config)
        print(f"\nRunning on: {device}")

    train_dataset, test_dataset = get_cifar10_datasets()
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
    )

    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        sampler=test_sampler,
        num_workers=config.num_workers,
    )

    model = ResNet18(num_classes=10)
    model.to(device)

    # Wrap the model with DDP
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False) 

    if rank == 0:
        writer = SummaryWriter(log_dir="runs/resnet18")
        dummy = torch.randn(1, 3, 32, 32).to(device)
        writer.add_graph(model.module, dummy)
        
        images, labels = next(iter(train_loader))
        writer.add_images("Training Images",images[:16])
    else:
        writer = None

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.epochs)

    # print("\nRunning PyTorch Profiler...")
    # profile_training(model, train_loader, loss_fn, optimizer, scaler, use_amp, device)
    # print("Profiler completed.")
    
    # return
    
    if rank == 0:
        print("\n" + "=" * 50)
        print("Training")
        print("=" * 50)
    
    best_accuracy = 0.0

    for epoch in range(config.epochs):

        train_sampler.set_epoch(epoch)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        train_loss = train_one_epoch_ddp(model, train_loader, loss_fn, optimizer, scaler, use_amp, device)
        
        if device.type == "cuda":
            torch.cuda.synchronize()

        train_time = time.perf_counter() - start_time

        if rank == 0:
            writer.add_scalar("Loss/Train", train_loss, epoch)
        
        val_loss, val_acc = validate_ddp(model, test_loader, loss_fn, device)
        
        if rank == 0:
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Accuracy", val_acc, epoch)

        current_lr = optimizer.param_groups[0]['lr']
        
        if rank == 0:
            writer.add_scalar("Current LR", current_lr, epoch)

        for name, param in model.named_parameters():
            if rank == 0:
                writer.add_histogram(name, param, epoch)

        scheduler.step()

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            if rank == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, best_accuracy, "checkpoints/best_model.pth")

        if rank == 0:
            print(f"Epoch {epoch+1}/{config.epochs}")
            print(f"Current LR: {current_lr:.4f}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Acc: {val_acc*100:.2f}%")
            print(f"Train Time: {train_time:.2f} seconds")
            print()

        if rank == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_accuracy, "checkpoints/last_checkpoint.pth")

    if rank == 0:
        print("\n" + "=" * 50)
        print("Training completed")
        print("=" * 50)
        writer.close()

    dist.destroy_process_group()
    
if __name__ == "__main__":
    main()