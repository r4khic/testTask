import argparse
from pathlib import Path
from typing import List, Dict

from inference import ViolencePredictor


def find_videos(directory: str) -> List[str]:
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}
    videos = []
    dir_path = Path(directory)

    if dir_path.exists():
        for file in dir_path.iterdir():
            if file.suffix.lower() in video_extensions:
                videos.append(str(file))

    return sorted(videos)


def print_results_table(results: List[Dict]) -> None:
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Video':<40} {'Prediction':<15} {'Probability':<12}")
    print("-" * 80)

    violence_count, non_violence_count, errors = 0, 0, 0

    for r in results:
        video_name = Path(r["video"]).name
        if len(video_name) > 38:
            video_name = video_name[:35] + "..."

        if "error" in r:
            print(f"{video_name:<40} {'ERROR':<15} {r['error']}")
            errors += 1
        else:
            print(f"{video_name:<40} {r['label']:<15} {r['probability']:.4f}")
            if r["label"] == "Violence":
                violence_count += 1
            else:
                non_violence_count += 1

    print("-" * 80)
    print(f"Total: {len(results)} videos")
    print(f"  Violence: {violence_count}")
    print(f"  Non-Violence: {non_violence_count}")
    if errors > 0:
        print(f"  Errors: {errors}")
    print("=" * 80)


def test_videos(model_path: str, videos_dir: str = None, video_files: List[str] = None, device: str = None) -> List[Dict]:
    all_videos = []

    if videos_dir:
        dir_videos = find_videos(videos_dir)
        all_videos.extend(dir_videos)
        print(f"Found {len(dir_videos)} videos in {videos_dir}")

    if video_files:
        all_videos.extend(video_files)

    if not all_videos:
        print("No videos to process!")
        return []

    all_videos = list(set(all_videos))
    print(f"\nTotal videos to process: {len(all_videos)}")
    print(f"\nLoading model from: {model_path}")

    predictor = ViolencePredictor(model_path, device)
    results = []

    for i, video_path in enumerate(all_videos, 1):
        print(f"\n[{i}/{len(all_videos)}] Processing: {Path(video_path).name}")
        try:
            label, prob, metadata = predictor.predict(video_path)
            indicator = "[!]" if label == "Violence" else "[ ]"
            print(f"  {indicator} {label} (confidence: {prob:.4f})")
            results.append({"video": video_path, "label": label, "probability": prob, "metadata": metadata})
        except Exception as e:
            print(f"  [X] Error: {e}")
            results.append({"video": video_path, "error": str(e)})

    return results


def main():
    parser = argparse.ArgumentParser(description="Test Violence Detection on Real Videos")
    parser.add_argument("--model", type=str, default="checkpoints/best_model")
    parser.add_argument("--videos-dir", type=str, default=None)
    parser.add_argument("--videos", type=str, nargs="+", default=None)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    model_config = Path(args.model).with_suffix(".json")
    model_weights = Path(args.model).with_suffix(".safetensors")

    if not model_config.exists() or not model_weights.exists():
        print(f"Model not found at: {args.model}")
        print("\nTrain the model first with:")
        print("  python train.py --data-dir <path_to_dataset>")
        return

    results = test_videos(
        model_path=args.model,
        videos_dir=args.videos_dir,
        video_files=args.videos,
        device=args.device
    )

    if results:
        print_results_table(results)


if __name__ == "__main__":
    main()
