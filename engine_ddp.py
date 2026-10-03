import torch
import torch.distributed as dist

def train_one_epoch_ddp(model, train_loader, loss_fn, optimizer, scaler, use_amp, device):
    
    model.train()

    # Accumulate loss as SUM over individual samples
    total_loss = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0, device=device)

    for step, (images, labels) in enumerate(train_loader):

        if step == 0:
            print("[Rank] First batch received", flush=True)

        images = images.to(device)
        labels = labels.to(device)

        if step == 0:
            print("[Rank] Data moved to GPU", flush=True)

        optimizer.zero_grad()

        if step == 0:
            print("[Rank] Starting forward", flush=True)

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = loss_fn(logits, labels)

        if step == 0:
            print("[Rank] Forward finished", flush=True)

        if step == 0:
            print("[Rank] Starting backward", flush=True)

        scaler.scale(loss).backward()

        if step == 0:
            print("[Rank] Backward finished", flush=True)

        scaler.step(optimizer)
        scaler.update()

        if step == 0:
            print("[Rank] Optimizer step finished", flush=True)

        # CrossEntropyLoss uses mean reduction by default.
        # Multiply by batch size to recover the total loss
        # contributed by this batch.
        total_loss += loss.detach() * labels.size(0)
        total_samples += labels.size(0)

    # Combine loss statistics from every rank.
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

    # Global sample-weighted mean loss
    epoch_loss = total_loss / total_samples

    return epoch_loss.item()


def validate_ddp(model, val_loader, loss_fn, device):

    model.eval()

    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device)
    total_samples = torch.tensor(0, device=device)

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            predictions = logits.argmax(dim=1)

            total_loss += loss.detach() * labels.size(0)
            total_correct += (predictions == labels).sum()
            total_samples += labels.size(0)

    # Combine metrics from all ranks
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

    val_loss = total_loss / total_samples
    val_acc = total_correct.float() / total_samples

    return val_loss.item(), val_acc.item()
