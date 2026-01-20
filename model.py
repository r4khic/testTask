import torch
import torch.nn as nn
from torchvision import models


class TimeDistributed(nn.Module):
    """Applies a module to each temporal slice of input (batch, time, C, H, W)."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, C, H, W = x.size()
        x = x.contiguous().view(batch_size * time_steps, C, H, W)
        x = self.module(x)
        x = x.contiguous().view(batch_size, time_steps, -1)
        return x


class MobileNetV2Features(nn.Module):
    """MobileNetV2 feature extractor (1280 features per frame)."""

    def __init__(self, freeze_layers: int = 100):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        for i, param in enumerate(self.features.parameters()):
            if i < freeze_layers:
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class ViolenceDetector(nn.Module):
    """MobileNetV2 + BiLSTM for violence detection."""

    def __init__(
        self,
        num_frames: int = 16,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        dropout: float = 0.3,
        freeze_cnn_layers: int = 100
    ):
        super().__init__()
        self.num_frames = num_frames

        cnn = MobileNetV2Features(freeze_layers=freeze_cnn_layers)
        self.time_distributed_cnn = TimeDistributed(cnn)

        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.time_distributed_cnn(x)
        lstm_out, _ = self.lstm(features)
        last_output = lstm_out[:, -1, :]
        return self.classifier(last_output)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (self.predict_proba(x) >= threshold).long()


def get_model(
    num_frames: int = 16,
    lstm_hidden_size: int = 256,
    lstm_num_layers: int = 2,
    dropout: float = 0.3,
    device: str = "cuda"
) -> ViolenceDetector:
    model = ViolenceDetector(
        num_frames=num_frames,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        dropout=dropout
    )
    return model.to(device)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = get_model(num_frames=16, device=device)

    dummy_input = torch.randn(2, 16, 3, 224, 224).to(device)
    print(f"Input: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output: {output.shape}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")
