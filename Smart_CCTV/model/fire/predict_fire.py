import torch
from torchvision import models, transforms
from PIL import Image
import yaml

# Load config
with open("fire/config/fire_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, cfg["num_classes"])
model.load_state_dict(torch.load("fire/checkpoint/fire_model.pt", map_location="cpu"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((cfg["img_size"], cfg["img_size"])),
    transforms.ToTensor()
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        predicted = output.argmax(dim=1).item()
    label = cfg["label_map"].get(str(predicted), str(predicted))
    print(f"Prediction: {label}")

# Example
predict("dataset/fire/preprocessed_frame/fire_01/frame_001.jpg")