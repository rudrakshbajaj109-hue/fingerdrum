# FingerDrum

FingerDrum turns your webcam into a virtual drum kit. Point your index fingers at the on-screen pads and play drum sounds with hand movement. It uses MediaPipe for hand tracking, OpenCV for the camera view and visuals, and Pygame for audio playback.

## Features

- Tracks up to two hands and detects index-finger movement.
- Eight interactive pads: Kick, Snare, Hi-Hat, Clap, Tom 1, Tom 2, Crash, and Ride.
- Neon pad, cursor, hit, ripple, and particle animations.
- Fullscreen webcam view with a live hand and FPS display.
- Acoustic drum samples included in `assets/kits/Acoustic`.

## Requirements

- Windows 10 or 11
- Python 3.12 recommended
- A webcam and audio output device

The hand landmark model is included at `assets/hand_landmarker.task`.

## Setup

Open PowerShell in the project folder and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, run the app with the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

From the project folder:

```powershell
.\.venv\Scripts\Activate.ps1
python .\src\main.py
```

The app opens the webcam view in fullscreen. Press **Esc** to exit. Allow camera access if Windows asks. Keep your hands visible in the camera, then move an index finger down toward a drum pad to trigger it.

## Performance

The app requests a 1280×720 camera feed at 60 FPS and limits the internal image size to keep tracking and effects responsive. Actual FPS depends on the webcam, lighting, and computer. If tracking is unreliable, improve the lighting and move your hand clearly into the camera view.

## Project structure

```text
assets/
  hand_landmarker.task
  kits/Acoustic/       Drum samples
src/
  audio/               Sample playback
  drum/                Pad geometry and hit detection
  effects/             Particle and post-processing effects
  ui/                  Pad, cursor, and animation rendering
  vision/              MediaPipe hand tracking
  main.py              Application entry point
```

## Credits

Made by **Rudraksh Bajaj**.
