# Video Scene Segmentation Counter

A Python-based tool that takes in a video file, processes frames using HSV color histograms and window-based comparison, and outputs the frame numbers and timestamps of every detected scene boundary.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

---

## Our Team

| Name | Role |
|---|---|
| Tyler Kilgore | Video Input / Preprocessing |
| Abraham Abebe | Feature Extraction / Window Processing |
| Andreas Paramo | Scene Detection Logic / Output |

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/TylerKilgoreUNT/video-scene-segmentation-g4.git
cd video-scene-segmentation-g4
```

### 2. Create a Virtual Environment

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
python main.py sample-5s.mp4
```

This plays the raw video then shows a preprocessed preview. Press **q** to close each window.

To use your own video:
```bash
python main.py path/to/your/video.mp4
```

---

## How It Works

1. Read video frames and convert to HSV
2. Compute color histograms per frame
3. Compare groups of frames using window-based aggregation
4. Detect large histogram differences as scene boundaries

---

## Output

- Frame number of each detected scene change
- Timestamp of each detected scene change
- TBA more info to come