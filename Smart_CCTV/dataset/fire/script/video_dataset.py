import os
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

class FireVideoDataset(Dataset):
    def __init__(self, video_dir, clip_len=16, transform=None):
        self.video_dir = video_dir
        self.clip_len = clip_len
        self.transform = transform or Compose([
            Resize((128, 128)),
            CenterCrop(112),
            ToTensor(),
            Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []
        for fname in sorted(os.listdir(self.video_dir)):
            if fname.endswith(".mp4"):
                label = fname.split("_")[0]  # e.g., "building"
                path = os.path.join(self.video_dir, fname)
                samples.append((path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
        total_frames = frames.shape[0]

        if total_frames < self.clip_len:
            raise ValueError(f"Video too short: {video_path}")

        start = torch.randint(0, total_frames - self.clip_len, (1,)).item()
        clip = frames[start:start + self.clip_len]  # shape: (clip_len, H, W, 3)

        clip = clip.permute(0, 3, 1, 2)  # (T, C, H, W)
        clip = torch.stack([self.transform(f) for f in clip])  # apply transform to each frame

        label_map = {"building": 0, "house": 1, "smoke": 2}
        return clip, label_map[label]