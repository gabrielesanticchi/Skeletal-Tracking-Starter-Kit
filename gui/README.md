# FIFA Skeletal Tracking - Synchronized Video Viewer

A PyQt6-based GUI for synchronized playback of original videos with 3D pose animations and pitch tracking visualizations.

## Overview

This GUI provides simple synchronized MP4 playback:
- **Original Video** (left panel)
- **3D Pose Animation** (top-right panel)
- **Pitch Tracking** (bottom-right panel)

**Key Features:**
- Works with any combination (1, 2, or 3 videos)
- Simple frame-index synchronization (no complex FPS calculations)
- Fixed 25 FPS playback timing
- Adjustable frame size (0.5x to 2.0x) for zooming
- Load videos on-demand with file browser
- No errors for missing animations

## Installation

The GUI uses a separate virtual environment (`.venv_gui`) to avoid conflicts:

```bash
# Environment already created with dependencies installed
# Just activate it:
source .venv_gui/bin/activate
```

## Quick Start

```bash
# Activate GUI environment
source .venv_gui/bin/activate

# Run with a sequence (loads only original video)
python gui/main.py --sequence ARG_CRO_220001
```

**In the GUI:**
1. Original video loads automatically
2. Click "📂 Load 3D Poses" to browse for pose animation
3. Click "📂 Load Pitch Tracking" to browse for tracking animation
4. Use controls to play/pause, seek, adjust speed

## Controls

### Playback Controls
- **Play/Pause**: Space key or button
- **Timeline Slider**: Click to seek to any frame
- **Step Forward/Back**: Left/Right arrow keys or buttons
- **Reset**: Home key or button
- **Jump to End**: End key

### Viewing Controls
- **Speed Control**: Up/Down arrows or spinbox (0.1x - 3.0x)
- **Frame Size**: Spinbox (0.5x - 2.0x) for zooming in/out
- **Fullscreen**: F key

### Loading Controls
- **Load 3D Poses**: Button to browse and load .mp4 animation
- **Load Pitch Tracking**: Button to browse and load .mp4 animation
- **Sequence Dropdown**: Switch between sequences

## Loading Animations

### Option 1: Use Load Buttons (Recommended)

The easiest way to use the GUI:

1. Launch: `python gui/main.py --sequence ARG_CRO_220001`
2. Click "📂 Load 3D Poses" or "📂 Load Pitch Tracking"
3. Browse to your .mp4 file
4. GUI validates frame count and loads video

**Benefits:**
- No need to pre-generate animations
- Load any .mp4 file from anywhere
- Mix and match different preprocessing outputs
- Test different visualizations

### Option 2: Auto-Load from Standard Location

If animations are in `results/SMPL/` with standard names, they load automatically:

```bash
# Generate animations
source .venv/bin/activate
python scripts/visualization/animate_smpl_poses.py --sequence ARG_CRO_220001 --fps 25
python scripts/visualization/visualize_smpl_poses.py --sequence ARG_CRO_220001 --pitch-view --animate-pitch --fps 25

# Launch GUI (finds animations automatically)
source .venv_gui/bin/activate
python gui/main.py --sequence ARG_CRO_220001
```

## Generating Animations

**Important:** All animations must have the same frame count as the original video!

### Manual Generation

```bash
# Activate main environment
source .venv/bin/activate

# Generate 3D poses animation (all frames, all 22 players)
python scripts/visualization/animate_smpl_poses.py \
    --sequence ARG_CRO_220001 \
    --fps 25 \
    --show-labels \
    --num-subjects 22

# Generate pitch tracking animation (all frames, all 22 players)
python scripts/visualization/visualize_smpl_poses.py \
    --sequence ARG_CRO_220001 \
    --pitch-view \
    --animate-pitch \
    --fps 25 \
    --trail-length 50 \
    --num-subjects 22
```

Animations are saved to `results/SMPL/` directory.

### Automated Generation

```bash
# Single sequence
python scripts/generate_animations_for_gui.py --sequence ARG_CRO_220001

# Multiple sequences
python scripts/generate_animations_for_gui.py --sequences ARG_CRO_220001 ARG_CRO_220954

# Quick test (first 200 frames only)
python scripts/generate_animations_for_gui.py --sequence ARG_CRO_220001 --end-frame 200
```

## Flexible Loading

The GUI works with any combination of videos:

- ✅ **Original only**: Basic video playback
- ✅ **Original + Poses**: Compare video with 3D reconstruction
- ✅ **Original + Tracking**: Check player positions on pitch
- ✅ **All three**: Full synchronized visualization

Missing animations? No problem! Just click the Load buttons.

## Quality Checking Workflow

### Full Sequence Validation

1. Load sequence in GUI
2. Play through at normal speed
3. Look for:
   - Tracking failures (players disappearing)
   - Pose estimation errors (unnatural joint positions)
   - Position mismatches between video and tracking

### Frame-by-Frame Analysis

1. Use Step Forward/Backward buttons
2. Check specific frames with potential errors
3. Verify 3D poses match original video
4. Use Frame Size control to zoom in for details

### Adjusting View

1. **Frame Size**: Use 0.5x to 2.0x to zoom in/out for detailed inspection
2. **Speed**: Use 0.1x to 3.0x for slow-motion analysis
3. **Fullscreen**: Press F key for maximum visibility

## Keyboard Shortcuts

- **Space**: Play/Pause
- **Left/Right Arrow**: Step backward/forward one frame
- **Up/Down Arrow**: Increase/decrease playback speed
- **Home**: Jump to start
- **End**: Jump to end
- **F**: Toggle fullscreen

## GUI Layout

```
┌─────────────────────┬─────────────────────┐
│                     │                     │
│  Original Video     │  3D Poses           │
│  (Master)           │  Animation          │
│                     │                     │
├─────────────────────┼─────────────────────┤
│                     │                     │
│  Control Panel      │  Pitch Tracking     │
│  - Play/Pause       │  Animation          │
│  - Speed/Size       │                     │
│  - Load Buttons     │                     │
└─────────────────────┴─────────────────────┘
```

## Technical Details

### Synchronization Method
- Uses direct frame index (not time-based)
- All videos seek to same frame number
- Assumes all videos have same frame count
- Threading locks prevent race conditions

### Performance Optimizations
- FastTransformation for faster rendering
- No FPS calculations per frame
- Minimal overhead
- Fixed 25 FPS playback timer (configurable via `PLAYBACK_FPS` constant in `main.py`)

### Architecture

**Modules:**
- `gui/main.py`: Main application window
- `gui/widgets/video_panel.py`: Video display widget
- `gui/widgets/control_panel.py`: Playback control widget
- `gui/utils/video_loader.py`: Video loading utilities
- `gui/utils/sync_manager.py`: Multi-video synchronization

**Code Structure:**
```
gui/
├── main.py (485 lines)
│   ├── SynchronizedViewer class
│   ├── _load_sequence() - loads original video
│   ├── _load_poses_file() - file browser for poses
│   ├── _load_tracking_file() - file browser for tracking
│   └── _load_additional_video() - validates & loads video
├── widgets/
│   ├── control_panel.py (234 lines)
│   │   ├── Load buttons with signals
│   │   ├── Speed/scale controls
│   │   └── Sequence selector
│   └── video_panel.py (110 lines)
│       ├── Frame display with scaling
│       └── Title updates
└── utils/
    ├── sync_manager.py (96 lines)
    │   ├── Simple frame-index sync
    │   └── Threading locks
    └── video_loader.py (87 lines)
        └── OpenCV video loading
```

## Troubleshooting

### GUI is Slow

The GUI has been optimized for speed:
- ✓ No complex FPS calculations
- ✓ Fast frame transformation
- ✓ Direct frame index synchronization
- ✓ Minimal overhead

If still slow:
1. Reduce Frame Size (try 0.7x or 0.5x)
2. Close other applications
3. Reduce playback speed

### Videos Not Synchronized

Ensure all animations have the same frame count as the original video:

```bash
# Check frame counts
ffprobe -v error -select_streams v:0 \
  -show_entries stream=nb_frames \
  -of default=nokey=1:noprint_wrappers=1 video.mp4
```

All three videos must have identical frame counts for perfect sync!

### Frame Count Mismatch

If you load a video with different frame count:
- GUI shows warning dialog
- User can choose to continue anyway
- Videos will go out of sync at different rates
- Recommended: Regenerate animation with correct frame count

### Missing Animations

Missing animations are no longer a problem:
- Panels show "Click Load button →" message
- Use Load buttons to browse for .mp4 files
- No error messages or forced generation
- GUI works perfectly with just the original video

### Video Codec Issues

If videos don't play properly:
- Ensure ffmpeg is installed on your system
- Check that animations were generated successfully
- Verify file paths are correct
- Try different video codec (H.264 recommended)

## Requirements

- Python 3.12
- PyQt6
- OpenCV (opencv-python)
- NumPy

All dependencies are installed in `.venv_gui`.

## Examples

### Basic Usage

```bash
source .venv_gui/bin/activate
python gui/main.py --sequence ARG_CRO_220001
```

### Checking Pose Quality

1. Load a sequence
2. Use step forward/backward to examine specific frames
3. Compare original video with 3D poses to verify accuracy
4. Use Frame Size 1.5x or 2.0x to zoom in on details
5. Check pitch tracking to ensure proper player positioning

### Analyzing Full Clips

1. Generate full-length animations with all players
2. Load in GUI and play through the entire sequence
3. Use speed 0.5x to slow down complex movements
4. Verify all 22 players are tracked correctly throughout

### Testing Different Preprocessing

1. Generate multiple versions of animations
2. Load sequence in GUI
3. Use Load buttons to switch between different preprocessing outputs
4. Compare quality across different methods

## Performance Tips

1. **For quick testing**: Generate shorter animations (e.g., first 200 frames)
2. **For full validation**: Generate complete sequences with all players
3. **Playback speed**: Use 0.5x or slower for detailed analysis
4. **Frame stepping**: Use arrow keys for precise frame-by-frame checking
5. **Frame size**: Use 1.5x-2.0x for close inspection, 0.5x-0.7x for overview

## Notes

- The GUI uses a 25 FPS playback rate by default (configurable)
- Original video determines the master timeline (frame count)
- All three videos synchronized by frame index
- Load buttons allow flexibility in testing different outputs
- No animations required to start - load them when needed!
