import argparse
import json
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
from torchvision import transforms
from safetensors.torch import load_file

from model import ViolenceDetector


class ViolencePredictor:
    """Violence detection predictor for video files."""

    LABELS = {0: "Non-Violence", 1: "Violence"}

    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        config_path = Path(model_path).with_suffix(".json")
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.num_frames = self.config["num_frames"]
        self.frame_size = self.config.get("frame_size", 224)

        self.model = ViolenceDetector(
            num_frames=self.config["num_frames"],
            lstm_hidden_size=self.config["lstm_hidden_size"],
            lstm_num_layers=self.config["lstm_num_layers"],
            dropout=self.config["dropout"]
        )

        weights_path = Path(model_path).with_suffix(".safetensors")
        state_dict = load_file(str(weights_path))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"Model loaded from: {model_path}")
        print(f"Device: {self.device}")
        print(f"Config: {self.config}")

    def _extract_frames(self, video_path: str) -> Tuple[np.ndarray, float, int]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0:
            cap.release()
            raise RuntimeError(f"Video has no frames: {video_path}")

        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                frames.append(frame)
            elif frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8))

        cap.release()
        return np.array(frames), fps, total_frames

    def _preprocess(self, frames: np.ndarray) -> torch.Tensor:
        transformed = [self.transform(frame) for frame in frames]
        tensor = torch.stack(transformed).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, video_path: str) -> Tuple[str, float, dict]:
        frames, fps, total_frames = self._extract_frames(video_path)
        input_tensor = self._preprocess(frames)

        with torch.no_grad():
            logits = self.model(input_tensor)
            prob = torch.sigmoid(logits).item()

        prediction = 1 if prob >= 0.5 else 0
        label = self.LABELS[prediction]

        metadata = {
            "video_path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "duration_sec": total_frames / fps if fps > 0 else 0,
            "sampled_frames": self.num_frames,
            "violence_probability": prob,
            "prediction": prediction,
            "label": label
        }

        return label, prob, metadata


def main():
    parser = argparse.ArgumentParser(description="Run violence detection on video files")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--video", type=str, nargs="+", required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    predictor = ViolencePredictor(args.model, args.device)
    results = []

    print("\n" + "=" * 60)

    for video_path in args.video:
        print(f"\nProcessing: {video_path}")
        try:
            label, prob, metadata = predictor.predict(video_path)
            print(f"  Prediction: {label}")
            print(f"  Violence probability: {prob:.4f}")
            print(f"  Duration: {metadata['duration_sec']:.1f}s")
            results.append({"video": video_path, "label": label, "probability": prob, "metadata": metadata})
        except Exception as e:
            print(f"  Error: {e}")
            results.append({"video": video_path, "error": str(e)})

    print("\n" + "=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
