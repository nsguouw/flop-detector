# NBA Flop Detector — Prototype

A combined pose-extraction + labeling pipeline for building an NBA flop
detection dataset. Uses MediaPipe Pose to extract body mechanics from video
clips, computes flop-relevant features, and provides an interactive CLI
labeling tool to build training data.

---

## Project structure

```
flop_detector/
├── extractor.py      # Pose estimation + feature extraction
├── labeler.py        # Interactive labeling tool
├── requirements.txt  # Python dependencies
├── clips/            # Put your video clips here
├── data/             # Dataset JSON saved here
│   └── dataset.json
└── output/           # Reserved for future outputs (annotated videos etc.)
```

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> MediaPipe requires Python 3.8–3.11. If you're on 3.12+, use 3.11 via pyenv.

---

## Getting video clips

Good free sources for NBA clips:

- **YouTube** — search "NBA flop compilation", "NBA foul baiting", or specific
  player names. Download with `yt-dlp`:
  ```bash
  pip install yt-dlp
  yt-dlp -o "clips/%(title)s.%(ext)s" "<youtube_url>"
  ```
- **Internet Archive** — archive.org has historical NBA broadcasts
- **NBA stats site** — some play clips are embeddable/downloadable

**Recommended clip length:** 2–8 seconds. Trim to just the contact moment.
A tool like `ffmpeg` makes this easy:
```bash
# Trim from 0:12 to 0:17 of a longer video
ffmpeg -i input.mp4 -ss 00:00:12 -to 00:00:17 -c copy clips/clip_001.mp4
```

**Aim for balance:** try to get roughly equal flop vs. non-flop clips.
A good starting dataset is 50–100 clips of each class.

---

## Running the labeler

```bash
# Label all clips in the clips/ directory
python labeler.py

# Custom paths
python labeler.py --clips_dir my_clips/ --dataset data/my_dataset.json

# View dataset statistics
python labeler.py --stats
```

### Labeling controls

| Key | Action         |
|-----|----------------|
| `f` | Flop           |
| `n` | Not a flop     |
| `u` | Unsure/borderline |
| `s` | Skip this clip |
| `q` | Quit and save  |

The dataset is saved after every label — safe to quit and resume anytime.

---

## Running the extractor standalone

```bash
python extractor.py clips/my_clip.mp4
```

Prints extracted features and a heuristic flop score (0–1).

---

## Features extracted per clip

| Feature | Description |
|---|---|
| `head_snap_max` | Peak velocity of nose vs. hips — key flop signal |
| `fall_rate_max` | Fastest downward movement of center of gravity |
| `trunk_delta_max` | Maximum trunk angle change per frame (whipping) |
| `nose_velocity_max` | Absolute peak head speed |
| `min_nose_y` | Lowest point of head (how far did they go down) |
| `*_mean`, `*_std` | Mean and standard deviation of each feature |

---

## What's next (after labeling ~100+ clips)

Once you have enough labeled data, the next step is training a simple
classifier. A good starting point:

```python
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

with open("data/dataset.json") as f:
    dataset = json.load(f)

# Filter out unsure and unlabeled
labeled = [d for d in dataset if d["label"] in ("flop", "not_flop")]

feature_keys = [k for k in labeled[0]["clip_features"]]
X = np.array([[d["clip_features"].get(k, 0) for k in feature_keys] for d in labeled])
y = np.array([1 if d["label"] == "flop" else 0 for d in labeled])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print(classification_report(y_test, clf.predict(X_test)))
```

---

## Known limitations

- MediaPipe works best on well-lit, unoccluded shots of a single player.
  Wide-angle broadcast shots with multiple players are harder.
- Contact force cannot be directly measured from video — the model infers it
  from the *reaction*, which is exactly what floppers exploit.
- Ground truth labels are inherently subjective. When unsure, label as
  `unsure` rather than forcing a call — you can filter these out at training time.
