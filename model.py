"""
import torch
import torch.nn as nn


class LipReadingModel(nn.Module):
    def __init__(self, vocab_size=1000):
        super(LipReadingModel, self).__init__()

        # 1. Frontend: 3D CNN for Spatiotemporal Features
        self.frontend = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        # 2. Flatten for Sequence Modeling
        # Assuming input frames 100x50, after pooling, dims change
        self.fc_visual = nn.Linear(32 * 12 * 25, 256)

        # 3. Sequence Modeling: GRU
        self.gru = nn.GRU(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)

        # 4. Classification
        self.fc_out = nn.Linear(128 * 2, vocab_size)

    def forward(self, x):
        # Input: (Batch, Channel, Time, Height, Width)
        b, c, t, h, w = x.size()

        # Frontend
        x = self.frontend(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b, t, -1)  # Flatten spatial dims

        # Projection
        x = self.fc_visual(x)

        # Sequence Modeling
        x, _ = self.gru(x)

        # Classification
        out = self.fc_out(x)
        return out


def get_model():
    model = LipReadingModel()
    model.eval()
    return model
"""

import torch
import torch.nn as nn


class LipReadingModel(nn.Module):
    def __init__(self, num_classes=40):  # 40 chars/tokens + 1 blank for CTC
        super(LipReadingModel, self).__init__()

        # --- LAYER 1: 3D-CNN Feature Encoder ---
        # Input: (Batch, Channel=1, Time, Height, Width) -> Grayscale
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        # --- LAYER 2: Sequence Processor (Bi-GRU/LSTM) ---
        # Calculate input size after 3D CNN flattening
        # Assuming input frame 100x50 -> downsampled twice -> approx 25x12
        self.rnn_input_size = 64 * 12 * 25

        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # --- LAYER 3: CTC Transcription Layer ---
        # Maps RNN output to Character Classes
        self.fc = nn.Linear(256 * 2, num_classes + 1)  # +1 for CTC Blank Token

    def forward(self, x):
        # x: (Batch, 1, Time, 50, 100)
        b, c, t, h, w = x.size()

        # 1. 3D-CNN Feature Extraction
        x = self.conv3d(x)  # -> (B, 64, T, H', W')

        # 2. Reshape for RNN (Flatten spatial dimensions, keep Time)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # -> (B, T, 64, H', W')
        b, t, c, h, w = x.size()
        x = x.view(b, t, -1)  # -> (B, T, Features)

        # 3. Bi-LSTM Processing
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # -> (B, T, 512)

        # 4. CTC Logits
        x = self.fc(x)  # -> (B, T, NumClasses)

        # Returns Log-Probabilities for CTC Loss
        return x.log_softmax(dim=2)


def get_model():
    model = LipReadingModel()
    model.eval()
    return model