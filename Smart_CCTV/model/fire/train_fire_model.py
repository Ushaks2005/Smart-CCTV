import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import yaml

# Load config
with open("fire/config/fire_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Data
transform = transforms.Compose([
    transforms.Resize((cfg["img_size"], cfg["img_size"])),
    transforms.ToTensor()
])
train_data = datasets.ImageFolder(cfg["train_dir"], transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, cfg["num_classes"])
model = model.to(device)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])

for epoch in range(cfg["epochs"]):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{cfg['epochs']} - Loss: {total_loss:.4f}")

# Save model
os.makedirs("fire/checkpoint", exist_ok=True)
torch.save(model.state_dict(), "fire/checkpoint/fire_model.pt")
print("✅ fire_model.pt saved.")