# model/crime/train_crime_model.py
import os
import yaml
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def load_config(path="model/crime/config/crime_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_dataloaders(cfg):
    transform = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(cfg["train_dir"], transform=transform)
    val_ds = datasets.ImageFolder(cfg["val_dir"], transform=transform) if os.path.exists(cfg["val_dir"]) else None

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True) if val_ds else None
    return train_loader, val_loader, train_ds.classes

def build_model(num_classes, pretrained=True, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device), device

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    if loader is None:
        return None, None
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def main():
    cfg = load_config()
    train_loader, val_loader, classes = make_dataloaders(cfg)
    model, device = build_model(cfg["num_classes"], pretrained=cfg.get("pretrained", True))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    best_val_acc = 0.0
    log_dir = cfg.get("log_dir", "model/crime/logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        line = f"Epoch {epoch}/{cfg['epochs']} | train_loss: {train_loss:.4f} train_acc: {train_acc:.4f}"
        if val_loss is not None:
            line += f" | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}"
        print(line)
        with open(log_file, "a") as f:
            f.write(line + "\n")

        if val_acc and val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = cfg.get("checkpoint_path", "model/crime/checkpoints/crime_model.pt")
            save_checkpoint(model, ckpt_path)
            print(f"Saved best checkpoint to {ckpt_path}")

    # final save if no validation provided
    if not os.path.exists(cfg.get("checkpoint_path", "model/crime/checkpoints/crime_model.pt")):
        save_checkpoint(model, cfg.get("checkpoint_path", "model/crime/checkpoints/crime_model.pt"))
        print("Saved final checkpoint")

if __name__ == "__main__":
    main()