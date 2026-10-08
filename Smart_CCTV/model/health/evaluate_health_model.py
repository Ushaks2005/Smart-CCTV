# health/evaluate_health_model.py
import yaml
import torch
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

def load_cfg(path="health/config/health_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_model(cfg, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, cfg["num_classes"])
    model.load_state_dict(torch.load(cfg["checkpoint_path"], map_location=device))
    model.to(device).eval()
    return model, device

def make_loader(cfg):
    t = transforms.Compose([transforms.Resize((cfg["img_size"], cfg["img_size"])), transforms.ToTensor()])
    val_ds = datasets.ImageFolder(cfg["val_dir"], transform=t)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)
    return loader, val_ds.classes

def evaluate():
    cfg = load_cfg()
    model, device = load_model(cfg)
    loader, classes = make_loader(cfg)
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())
    print(classification_report(y_true, y_pred, target_names=classes))
    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate()