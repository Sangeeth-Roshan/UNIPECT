# UNIPECT — Uniform Inspection System

> A real-time, two-stage computer vision system that checks for school uniform compliance and ID card presence using a live webcam feed.

---

## What it does

UNIPECT uses two independently trained image classification models to enforce uniform compliance at entry points — no human checker required.

**Stage 1 — Uniform check:** The system captures 5 frames from the webcam and runs each through a MobileNet-based model trained on Teachable Machine. It averages the confidence scores across frames to reduce false positives from motion blur or lighting variation.

**Stage 2 — ID card check:** If the uniform check passes, a second model activates to verify whether the person is wearing a visible ID card. The final result is one of three states: fully compliant (green), uniform only — missing ID (red), or not in uniform (red).

Both checks run in a background thread so the GUI stays responsive throughout.

---

## Demo

| State | Result |
|---|---|
| Uniform + ID card | Green screen — "You are in uniform" + "You are wearing an ID card" |
| Uniform, no ID | Red screen — "But you aren't wearing an ID card!" |
| No uniform | Red screen — "You are not in uniform!" |

---

## Tech stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| TensorFlow / Keras | Model loading and inference |
| Teachable Machine | Model training (transfer learning on MobileNet) |
| OpenCV | Webcam capture and frame preprocessing |
| Tkinter + PIL | Desktop GUI and live video display |
| Threading | Non-blocking inference during UI updates |

---

## Project structure

```
UNIPECT/
├── models/
│   ├── main/
│   │   ├── keras_model.h5     # Uniform detection model
│   │   └── labels.txt         # Class labels for uniform model
│   └── id/
│       ├── keras_model.h5     # ID card detection model
│       └── labels.txt         # Class labels for ID model
├── mods/
│   └── ID.py                  # ID card check module
├── test/                      # Test images / scripts
├── root.py                    # Main GUI application (run this)
├── main.py                    # Headless CLI version
├── requirements.txt
└── .gitignore
```

---

## How it works

```
Webcam
  │
  ▼
Capture 5 frames
  │
  ▼
Resize to 224×224 → Normalize to [-1, 1]
  │
  ▼
Stage 1: Uniform model inference (avg. confidence across 5 frames)
  │
  ├── mean < 50%  →  NOT IN UNIFORM  →  Red UI
  │
  └── mean ≥ 50%  →  IN UNIFORM
                        │
                        ▼
                   Stage 2: ID card model inference (5 frames)
                        │
                        ├── mean < 50%  →  MISSING ID  →  Red UI
                        │
                        └── mean ≥ 50%  →  FULLY COMPLIANT  →  Green UI
```

The `CustomDepthwiseConv2D` wrapper silently drops the `groups` argument that Teachable Machine exports into the model config but that Keras does not support — allowing clean model loading without modifying the `.h5` file.

---

## Setup

**Prerequisites:** Python 3.9–3.11, a working webcam, pip.

```bash
# 1. Clone the repo
git clone https://github.com/Sangeeth-Roshan/UNIPECT.git
cd UNIPECT

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **Windows note:** If `opencv-python` fails to open the webcam, try installing `opencv-python-headless` instead.

> **macOS note:** Tkinter may require a separate install: `brew install python-tk`.

---

## Usage

### GUI mode (recommended)

```bash
python root.py
```

Click **Start Inspection** to begin. The system will run Stage 1, then Stage 2 automatically. Use **Check Again** to reset for the next person.

### Headless / CLI mode

```bash
python main.py
```

Runs the same two-stage check without a GUI. Outputs class names and confidence scores to the terminal.

---

## Training your own models

UNIPECT uses models trained with [Google Teachable Machine](https://teachablemachine.withgoogle.com/).

1. Go to Teachable Machine → Image Project → Standard image model.
2. Create two classes (e.g. `uniform` / `no_uniform`).
3. Train and export as **Keras** → download the `.h5` model and `labels.txt`.
4. Place under `models/main/` (uniform model) or `models/id/` (ID card model).

The models are not included in this repo due to file size. You will need to train your own on your institution's uniform.

---

## Author

**Sangeeth Roshan**
[github.com/Sangeeth-Roshan](https://github.com/Sangeeth-Roshan)
[linkedin.com/in/rsangeethroshan](http://linkedin.com/in/rsangeethroshan)

---
