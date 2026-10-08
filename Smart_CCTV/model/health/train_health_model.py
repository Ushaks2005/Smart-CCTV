# health/train_health_model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import yaml
from datetime import datetime

def load_cfg(path="health/config/health_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_loaders(cfg):
    t = transforms.Compose([transforms.Resize((cfg["img_size"], cfg["img_size"])), transforms.ToTensor()])
    train_ds = datasets.ImageFolder(cfg["train_dir"], transform=t)
    val_ds = datasets.ImageFolder(cfg["val_dir"], transform=t) if os.path.exists(cfg["val_dir"]) else None
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True) if val_ds else None
    return train_loader, val_loader, train_ds.classes

def build_model(num_classes, pretrained=True, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device), device

def save_ckpt(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def train():
    cfg = load_cfg()
    train_loader, val_loader, classes = make_loaders(cfg)
    model, device = build_model(cfg["num_classes"], pretrained=cfg.get("pretrained", True))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    os.makedirs(os.path.dirname(cfg["log_path"]), exist_ok=True)
    log_file = cfg["log_path"]

    best_val_acc = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
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
        train_loss = running_loss / total
        train_acc = correct / total

        val_loss = val_acc = None
        if val_loader:
            model.eval()
            v_run = 0.0
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    v_run += loss.item() * imgs.size(0)
                    preds = outputs.argmax(dim=1)
                    v_correct += (preds == labels).sum().item()
                    v_total += labels.size(0)
            val_loss = v_run / v_total
            val_acc = v_correct / v_total

        line = f"{datetime.now().isoformat()} E{epoch} train_loss:{train_loss:.4f} train_acc:{train_acc:.4f}"
        if val_loss is not None:
            line += f" val_loss:{val_loss:.4f} val_acc:{val_acc:.4f}"
        print(line)
        with open(log_file, "a") as f:
            f.write(line + "\n")

        if val_acc and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_ckpt(model, cfg["checkpoint_path"])
            print(f"Saved best checkpoint to {cfg['checkpoint_path']}")

    if not os.path.exists(cfg["checkpoint_path"]):
        save_ckpt(model, cfg["checkpoint_path"])
        print(f"Saved final checkpoint to {cfg['checkpoint_path']}")

if __name__ == "__main__":
    train()