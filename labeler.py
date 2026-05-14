"""
labeler.py
----------
Interactive CLI tool for labeling video clips as flop / not_flop / unsure.
Processes each clip through the pose extractor, shows the heuristic score,
and saves everything to a JSON dataset for later ML training.

Usage:
    python labeler.py --clips_dir clips/ --dataset data/dataset.json

Controls during labeling:
    f  = flop
    n  = not a flop
    u  = unsure / borderline
    s  = skip this clip
    q  = quit and save
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from extractor import process_clip, heuristic_flop_score

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

VALID_LABELS = {"flop", "not_flop", "unsure"}


def load_dataset(path: str) -> list:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []


def save_dataset(dataset: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"[dataset] saved {len(dataset)} clips → {path}")


def already_labeled(dataset: list, clip_path: str) -> bool:
    return any(d["clip_path"] == str(clip_path) for d in dataset)


# ---------------------------------------------------------------------------
# Video preview (shows first frame + pose overlay placeholder)
# ---------------------------------------------------------------------------

def show_clip_preview(clip_path: str, window_name: str = "Clip Preview"):
    """Play the clip in a cv2 window. Press any key to close."""
    if not CV2_AVAILABLE:
        print("[preview] cv2 not available — skipping visual preview")
        return

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"[preview] could not open {clip_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay = max(1, int(1000 / fps))

    print(f"  [preview] playing clip — press any key to stop")
    while True:
        ret, frame = cap.read()
        if not ret:
            # loop
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        cv2.putText(
            frame,
            "f=flop  n=not_flop  u=unsure  s=skip  q=quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key != 255:  # any key pressed
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Feature summary printer
# ---------------------------------------------------------------------------

def print_features(clip_features: dict, score: float):
    print("\n  ┌─ Extracted features ─────────────────────────────┐")
    important = [
        ("head_snap_max",      "Head snap (max)    "),
        ("fall_rate_max",      "Fall rate (max)    "),
        ("trunk_delta_max",    "Trunk whip (max)   "),
        ("nose_velocity_max",  "Nose velocity (max)"),
        ("min_nose_y",         "Min nose height    "),
    ]
    for key, label in important:
        val = clip_features.get(key, None)
        if val is not None:
            bar_len = int(min(abs(val) * 400, 20))
            bar = "█" * bar_len
            print(f"  │  {label}: {val:+.4f}  {bar}")
    print(f"  │")
    print(f"  │  Heuristic flop score: {score:.2f} / 1.00  {'⚠ HIGH' if score > 0.5 else ''}")
    print(f"  └──────────────────────────────────────────────────┘")


# ---------------------------------------------------------------------------
# Main labeling loop
# ---------------------------------------------------------------------------

def label_clip(clip_path: str, dataset: list, dataset_path: str, index: int, total: int) -> str:
    """
    Process and label a single clip. Returns the label string or 'skip'/'quit'.
    """
    print(f"\n{'='*60}")
    print(f"  Clip {index}/{total}: {Path(clip_path).name}")
    print(f"{'='*60}")

    # Process
    print("  Processing clip through pose estimator...")
    t0 = time.time()
    result = process_clip(clip_path)
    elapsed = time.time() - t0

    if not result["frame_features"]:
        print("  [warning] No pose landmarks detected in this clip — skipping")
        return "skip"

    print(f"  Extracted {len(result['frame_features'])} frames in {elapsed:.1f}s")
    score = heuristic_flop_score(result["clip_features"])
    print_features(result["clip_features"], score)

    # Preview
    show_choice = input("\n  Play video preview? [y/n]: ").strip().lower()
    if show_choice == "y":
        show_clip_preview(clip_path)

    # Label
    while True:
        print("\n  Label this clip:")
        print("    [f] flop       [n] not a flop      [u] unsure")
        print("    [s] skip       [q] quit and save")
        raw = input("  Your choice: ").strip().lower()

        if raw == "q":
            return "quit"
        if raw == "s":
            return "skip"
        if raw in ("f", "n", "u"):
            label_map = {"f": "flop", "n": "not_flop", "u": "unsure"}
            label = label_map[raw]

            notes = input(f"  Notes (optional, press enter to skip): ").strip()

            entry = {
                **result,
                "label":      label,
                "notes":      notes,
                "labeled_at": datetime.utcnow().isoformat(),
                "heuristic_score": score,
            }
            dataset.append(entry)
            save_dataset(dataset, dataset_path)
            print(f"  ✓ Labeled as: {label.upper()}")
            return label
        else:
            print("  Invalid input — please press f, n, u, s, or q")


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def print_stats(dataset: list):
    if not dataset:
        print("\n  Dataset is empty.")
        return
    from collections import Counter
    counts = Counter(d["label"] for d in dataset if d.get("label"))
    total  = sum(counts.values())
    print(f"\n  ── Dataset stats ──────────────────")
    for label, count in sorted(counts.items()):
        bar = "█" * count
        print(f"  {label:12s}: {count:4d}  {bar}")
    print(f"  {'total':12s}: {total:4d}")
    print(f"  ───────────────────────────────────")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NBA flop labeling tool")
    parser.add_argument(
        "--clips_dir",
        default="clips",
        help="Directory containing video clips (.mp4, .avi, .mov)",
    )
    parser.add_argument(
        "--dataset",
        default="data/dataset.json",
        help="Path to dataset JSON file (created if missing)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics and exit",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    if args.stats:
        print_stats(dataset)
        return

    # Collect clips
    clips_dir = Path(args.clips_dir)
    if not clips_dir.exists():
        print(f"[error] clips directory not found: {clips_dir}")
        print(f"  Create it and add video clips, then re-run.")
        sys.exit(1)

    extensions = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    all_clips  = sorted([p for p in clips_dir.iterdir() if p.suffix.lower() in extensions])

    if not all_clips:
        print(f"[error] No video clips found in {clips_dir}")
        print(f"  Supported formats: {', '.join(extensions)}")
        sys.exit(1)

    # Filter already-labeled
    pending = [c for c in all_clips if not already_labeled(dataset, str(c))]

    print(f"\n  NBA Flop Labeling Tool")
    print(f"  ──────────────────────")
    print(f"  Clips found:    {len(all_clips)}")
    print(f"  Already labeled:{len(all_clips) - len(pending)}")
    print(f"  To label:       {len(pending)}")
    print_stats(dataset)

    if not pending:
        print("\n  All clips are already labeled!")
        return

    input(f"\n  Press enter to begin labeling {len(pending)} clip(s)...")

    for i, clip_path in enumerate(pending, 1):
        result = label_clip(str(clip_path), dataset, args.dataset, i, len(pending))
        if result == "quit":
            print("\n  Quitting — dataset saved.")
            break

    print("\n  ── Final dataset stats ──")
    print_stats(dataset)
    print("\n  Done! Run the following to see stats anytime:")
    print(f"    python labeler.py --stats --dataset {args.dataset}")


if __name__ == "__main__":
    main()
