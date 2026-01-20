import random
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class VideoDataset(Dataset):
    """Dataset for loading violence/non-violence video clips."""

    VIOLENCE = 1
    NON_VIOLENCE = 0

    def __init__(
        self,
        data_dir: str,
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
        transform: Optional[transforms.Compose] = None,
        augment: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.augment = augment

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        violence_dir = self.data_dir / "Violence"
        non_violence_dir = self.data_dir / "NonViolence"
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

        if violence_dir.exists():
            for video_path in violence_dir.iterdir():
                if video_path.suffix.lower() in video_extensions:
                    self.samples.append((video_path, self.VIOLENCE))

        if non_violence_dir.exists():
            for video_path in non_violence_dir.iterdir():
                if video_path.suffix.lower() in video_extensions:
                    self.samples.append((video_path, self.NON_VIOLENCE))

        if len(self.samples) == 0:
            raise ValueError(
                f"No videos found in {self.data_dir}. "
                "Expected folders: Violence/ and NonViolence/"
            )

        print(f"Loaded {len(self.samples)} videos from {self.data_dir}")

    def _extract_frames(self, video_path: Path) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            raise RuntimeError(f"Video has no frames: {video_path}")

        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                if frames:
                    frame = frames[-1].copy()
                else:
                    frame = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))

            frames.append(frame)

        cap.release()
        return np.array(frames)

    def _augment_frames(self, frames: np.ndarray) -> np.ndarray:
        if random.random() > 0.5:
            frames = frames[:, :, ::-1, :].copy()
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            frames = np.clip(frames * factor, 0, 255).astype(np.uint8)
        return frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path, label = self.samples[idx]
        frames = self._extract_frames(video_path)

        if self.augment:
            frames = self._augment_frames(frames)

        transformed_frames = [self.transform(frame) for frame in frames]
        frames_tensor = torch.stack(transformed_frames)
        label_tensor = torch.tensor([label], dtype=torch.float32)

        return frames_tensor, label_tensor


def create_data_loaders(
    data_dir: str,
    num_frames: int = 16,
    frame_size: Tuple[int, int] = (224, 224),
    batch_size: int = 8,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    full_dataset = VideoDataset(
        data_dir=data_dir,
        num_frames=num_frames,
        frame_size=frame_size,
        augment=False
    )

    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"Train samples: {train_size}, Validation samples: {val_size}")
    return train_loader, val_loader
