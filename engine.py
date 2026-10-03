import torch

def train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, use_amp, device):
    model.train()
    running_loss = 0.0

    for step, (images, labels) in enumerate(train_loader):
        
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

def validate(model, val_loader, loss_fn, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device,non_blocking=True), labels.to(device,non_blocking=True)

            logits = model(images)
            loss = loss_fn(logits, labels)

            running_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            epoch_loss = running_loss / len(val_loader)
            accuracy = correct / total
            
    return epoch_loss, accuracy