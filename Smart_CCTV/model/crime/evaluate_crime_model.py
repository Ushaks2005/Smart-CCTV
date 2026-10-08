
import os
import yaml
import torch
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

def load_config(path="model/crime/config/crime_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_model(cfg, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, cfg["num_classes"])
    model.load_state_dict(torch.load(cfg.get("checkpoint_path", "model/crime/checkpoints/crime_model.pt"), map_location=device))
    model.to(device).eval()
    return model, device

def make_dataloader(cfg):
    transform = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
    ])
    val_ds = datasets.ImageFolder(cfg["val_dir"], transform=transform)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)
    return val_loader, val_ds.classes

def evaluate(cfg_path="model/crime/config/crime_config.yaml"):
    cfg = load_config(cfg_path)
    model, device = load_model(cfg)
    val_loader, classes = make_dataloader(cfg)

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    print(classification_report(y_true, y_pred, target_names=classes))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    evaluate()