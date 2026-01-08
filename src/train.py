
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import HMDBFramesDataset

from models.cnn_lstm import CNN_LSTM

DATASET_PATH = r"E:\Study\PhD\Deep Learning\Project\Project_HDMB_1\data\HDMB51_Final"

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = HMDBFramesDataset(DATASET_PATH, sequence_length=15)
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

model = CNN_LSTM(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    total_loss = 0

    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/10], Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "models/model.pth")
print("Training complete. Model saved.")
