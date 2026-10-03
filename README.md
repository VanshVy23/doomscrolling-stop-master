# Doomscroller Controller

Plays a video alarm in QuickTime Player whenever you look away from your screen. The video stops the moment you look back directly at the camera.

## Requirements

- macOS
- Python 3.x
- QuickTime Player (pre-installed on macOS)

## Install dependencies

```bash
pip install opencv-python mediapipe numpy
```

## Setup

Open `main.py` and update the video path on line 17:

```python
VIDEO_PATH = '/your/path/to/video.mp4'
```

If your webcam is not detected, change the camera index on the `cv2.VideoCapture` line:
- `0` → built-in camera
- `1` → first external camera

## Run

```bash
python3 main.py
```

## Usage

- Just sit in front of your webcam.
- Look away from the screen → video plays instantly with sound.
- Look directly back at the screen → video stops after ~1 second.
- Press `ESC` to quit.

## First run on macOS

macOS will ask for **Camera Access** and **Automation** (to control QuickTime) permissions. Click **Allow** on both prompts.
