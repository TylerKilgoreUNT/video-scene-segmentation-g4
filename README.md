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