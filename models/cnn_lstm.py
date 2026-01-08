import torch
import torch.nn as nn
from torchvision import models

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=5, hidden_size=256):
        super().__init__()

        cnn = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(cnn.children())[:-1])  # remove FC
        self.cnn_out = 512

        self.lstm = nn.LSTM(
            input_size=self.cnn_out,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)

        feats = self.cnn(x)
        feats = feats.view(B, T, -1)

        lstm_out, _ = self.lstm(feats)
        out = self.fc(lstm_out[:, -1, :])

        return out
