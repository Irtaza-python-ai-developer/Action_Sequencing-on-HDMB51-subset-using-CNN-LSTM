import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class HMDBFramesDataset(Dataset):
    def __init__(self, root_dir, sequence_length=15):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.samples = []

        self.labels = sorted(os.listdir(root_dir))
        self.label_map = {l: i for i, l in enumerate(self.labels)}

        for label in self.labels:
            class_path = os.path.join(root_dir, label)
            for seq in os.listdir(class_path):
                seq_path = os.path.join(class_path, seq)
                frames = sorted(os.listdir(seq_path))
                self.samples.append((seq_path, frames, label))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_path, frames, label = self.samples[idx]

        if len(frames) >= self.sequence_length:
            frames = frames[:self.sequence_length]
        else:
            frames = frames + [frames[-1]] * (self.sequence_length - len(frames))

        images = []
        for f in frames:
            img = Image.open(os.path.join(seq_path, f)).convert("RGB")
            images.append(self.transform(img))

        images = torch.stack(images)  # (15, 3, 224, 224)
        return images, self.label_map[label]
