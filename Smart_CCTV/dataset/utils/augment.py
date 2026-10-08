import torchvision.transforms as T

def get_augmentation_pipeline():
    return T.Compose([
        T.Resize((128, 128)),
        T.CenterCrop(112),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])