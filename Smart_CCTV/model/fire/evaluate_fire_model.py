import torch
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report
import yaml

# Load config
with open("fire/config/fire_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Data
transform = transforms.Compose([
    transforms.Resize((cfg["img_size"], cfg["img_size"])),
    transforms.ToTensor()
])
val_data = datasets.ImageFolder(cfg["val_dir"], transform=transform)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg["batch_size"], shuffle=False)

# Model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, cfg["num_classes"])
model.load_state_dict(torch.load("fire/checkpoint/fire_model.pt", map_location="cpu"))
model.eval()

# Evaluation
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

print(classification_report(y_true, y_pred, target_names=cfg["class_names"]))