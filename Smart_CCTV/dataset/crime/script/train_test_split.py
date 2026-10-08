from video_dataset import CrimeVideoDataset
from torch.utils.data import DataLoader

dataset = CrimeVideoDataset("dataset/crime/raw_videos", clip_len=16)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for clips, labels in loader:
    print(clips.shape)  # (B, T, C, H, W)
    print(labels)       # tensor of class indices