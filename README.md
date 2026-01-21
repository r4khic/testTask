# Violence Detection in Videos

PyTorch model for binary video classification: violence vs non-violence.

**Architecture:** MobileNetV2 (pretrained) + Bidirectional LSTM
**Accuracy:** 96.25% on validation
**Format:** .safetensors

## Architecture

```
Video (16 frames, 224x224)
         │
         ▼
┌─────────────────────────────┐
│  TimeDistributed MobileNetV2 │  ← Pretrained ImageNet, frozen layers
│  Output: (batch, 16, 1280)   │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│    Bidirectional LSTM        │  ← 2 layers, 256 hidden
│    Output: (batch, 512)      │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│    FC: 512 → 256 → 64 → 1    │
│    + Dropout (0.3)           │
└─────────────────────────────┘
         │
         ▼
    Violence / Non-Violence
```

## Dataset

[Real Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)

- 1000 violence videos
- 1000 non-violence videos

Structure:
```
data/
├── Violence/
└── NonViolence/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python train.py --data-dir data/ --batch-size 4 --epochs 50
```

Parameters:
| Parameter | Default |
|-----------|---------|
| `--num-frames` | 16 |
| `--batch-size` | 8 |
| `--epochs` | 50 |
| `--lr` | 1e-4 |
| `--patience` | 7 |

Output:
- `checkpoints/best_model.safetensors`
- `checkpoints/best_model.json`

## Inference

```bash
# Single video
python inference.py --model checkpoints/best_model --video video.mp4

# Multiple videos
python inference.py --model checkpoints/best_model --video video1.mp4 video2.mp4

# Directory
python test_real_videos.py --model checkpoints/best_model --videos-dir test_videos/
```

Python API:
```python
from inference import ViolencePredictor

predictor = ViolencePredictor("checkpoints/best_model")
label, probability, metadata = predictor.predict("video.mp4")
```

## Results

### Validation
- **Best accuracy:** 96.25%
- **Training time:** ~74 minutes (GTX 1660 Ti)
- **Early stopping:** epoch 12

### Real Videos Test
| Video | Prediction | Probability |
|-------|------------|-------------|
| Walking (Google Maps) | Non-Violence | 0.05 |
| People talking | Non-Violence | 0.05 |
| Street fight | Violence | 0.52 |

## Project Structure

```
├── model.py              # MobileNetV2 + BiLSTM
├── dataset.py            # VideoDataset
├── train.py              # Training with early stopping
├── inference.py          # ViolencePredictor
├── test_real_videos.py   # Batch testing
├── requirements.txt
├── checkpoints/
│   ├── best_model.safetensors      # Trained model
│   ├── best_model.json
│   ├── mobilenet_v2_pretrained.pth # Pretrained backbone
│   └── training_history.json
└── README.md
```

## Pretrained Model

Fine-tuning base: **MobileNetV2** (ImageNet weights)
- File: `checkpoints/mobilenet_v2_pretrained.pth`
- Source: torchvision.models

## References

- [Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
- [MobileNet Bi-LSTM Reference](https://www.kaggle.com/code/abduulrahmankhalid/real-time-violence-detection-mobilenet-bi-lstm)
