# PyQt GUI Requirements for Synchronized Video/Pose/Tracking Viewer

## 📋 Overview

This document outlines the requirements for building a PyQt-based GUI application that synchronizes and displays:
1. **Original video** (left panel)
2. **3D pose animation** (top-right panel) 
3. **Pitch tracking animation** (bottom-right panel)

All three components must be synchronized to play at the same frame rate and duration for comprehensive analysis.

## 🎯 Core Requirements

### 1. **Multi-Panel Layout**
```
┌─────────────────┬─────────────────┐
│                 │                 │
│   Original      │   3D Poses      │
│   Video         │   Animation     │
│   (.mp4)        │   (.mp4)        │
│                 │                 │
├─────────────────┼─────────────────┤
│   Controls      │   Pitch         │
│   (Play/Pause/  │   Tracking      │
│   Seek/Speed)   │   Animation     │
│                 │   (.mp4)        │
└─────────────────┴─────────────────┘
```

### 2. **Video Synchronization**
- **Primary Source**: Original video determines the master timeline
- **Secondary Sources**: Pose and tracking animations adjust to match video duration
- **Frame-Perfect Sync**: All three videos must display the exact same temporal moment
- **Speed Control**: Unified playback speed control affects all three panels simultaneously

### 3. **Input Requirements**
- **Video File**: Original sequence video (e.g., `data/videos/train_data/ARG_CRO_220001.mp4`)
- **Pose Animation**: Generated MP4 from `animate_smpl_poses.py` (50 fps default)
- **Tracking Animation**: Generated MP4 from `visualize_smpl_poses.py --animate-pitch` (50 fps default)

## 🔧 Technical Specifications

### **Video Processing Requirements**
```python
# Required libraries
PyQt5 or PyQt6          # GUI framework
opencv-python           # Video processing
numpy                   # Data handling
matplotlib              # For any additional plotting needs
```

### **Synchronization Logic**
1. **Duration Matching**: 
   - Load all three videos and determine their durations
   - If durations don't match, adjust playback speed of pose/tracking to match original video
   - Formula: `adjusted_fps = original_fps * (pose_duration / video_duration)`

2. **Frame Synchronization**:
   - Use video frame number as master timeline
   - Calculate corresponding frame in pose/tracking animations
   - Handle frame rate differences automatically

### **GUI Components**

#### **Main Window** (`QMainWindow`)
- **Title**: "FIFA Skeletal Tracking Synchronized Viewer"
- **Size**: Minimum 1400x900 pixels
- **Layout**: Grid layout with 2x2 arrangement

#### **Video Panels** (`QLabel` or `QVideoWidget`)
- **Original Video Panel**: 
  - Position: Top-left (largest panel)
  - Size: 60% of window width, 70% of window height
  - Features: Display original sequence video

- **3D Poses Panel**:
  - Position: Top-right
  - Size: 40% of window width, 35% of window height
  - Features: Display 3D pose animation

- **Pitch Tracking Panel**:
  - Position: Bottom-right
  - Size: 40% of window width, 35% of window height
  - Features: Display pitch tracking animation

#### **Control Panel** (`QWidget`)
- **Position**: Bottom-left
- **Size**: 60% of window width, 30% of window height
- **Components**:
  - Play/Pause button (`QPushButton`)
  - Timeline slider (`QSlider`)
  - Speed control (`QDoubleSpinBox`: 0.1x to 3.0x)
  - Frame counter (`QLabel`)
  - Duration display (`QLabel`)
  - Sequence info (`QLabel`)

## 📁 File Structure

```
gui/
├── main.py                     # Main application entry point
├── widgets/
│   ├── __init__.py
│   ├── video_panel.py          # Custom video display widget
│   ├── control_panel.py        # Playback controls widget
│   └── sync_manager.py         # Video synchronization logic
├── utils/
│   ├── __init__.py
│   ├── video_loader.py         # Video file loading and validation
│   └── sync_calculator.py      # Duration/FPS synchronization calculations
└── requirements.txt            # Python dependencies
```

## 🚀 Implementation Steps

### **Phase 1: Basic GUI Setup**
1. Create main window with 2x2 grid layout
2. Implement basic video loading for original video
3. Add play/pause/seek controls
4. Test with single video playback

### **Phase 2: Multi-Video Support**
1. Extend video loading to handle 3 simultaneous videos
2. Implement frame synchronization logic
3. Add speed control affecting all videos
4. Test with pose and tracking animations

### **Phase 3: Advanced Features**
1. Add sequence selection dropdown
2. Implement automatic animation generation if missing
3. Add export functionality for synchronized clips
4. Performance optimization for smooth playback

## 📊 Data Flow

### **Input Processing**
```python
# 1. Load sequence data
sequence_name = "ARG_CRO_220001"
video_path = f"data/videos/train_data/{sequence_name}.mp4"

# 2. Generate animations if not exist
pose_animation = f"results/SMPL/{sequence_name}_poses_animation.mp4"
tracking_animation = f"results/SMPL/{sequence_name}_pitch_tracking.mp4"

# 3. Validate synchronization
video_duration = get_video_duration(video_path)
pose_duration = get_video_duration(pose_animation)
tracking_duration = get_video_duration(tracking_animation)

# 4. Calculate sync parameters
pose_speed_factor = pose_duration / video_duration
tracking_speed_factor = tracking_duration / video_duration
```

### **Playback Synchronization**
```python
# Master timeline from original video
current_frame = video_player.get_current_frame()
video_fps = video_player.get_fps()
current_time = current_frame / video_fps

# Synchronized frame calculation
pose_frame = int(current_time * pose_fps * pose_speed_factor)
tracking_frame = int(current_time * tracking_fps * tracking_speed_factor)

# Update all displays
video_player.seek_to_frame(current_frame)
pose_player.seek_to_frame(pose_frame)
tracking_player.seek_to_frame(tracking_frame)
```

## 🎮 User Interface Features

### **Menu Bar**
- **File Menu**:
  - Open Sequence (loads video + generates animations if needed)
  - Export Synchronized Clip
  - Exit

- **View Menu**:
  - Toggle Pose Panel
  - Toggle Tracking Panel
  - Fullscreen Mode

- **Tools Menu**:
  - Generate Missing Animations
  - Sync Settings
  - Performance Settings

### **Keyboard Shortcuts**
- `Space`: Play/Pause
- `Left/Right Arrow`: Frame-by-frame navigation
- `Up/Down Arrow`: Speed adjustment
- `Home/End`: Jump to start/end
- `F`: Toggle fullscreen
- `1/2/3`: Focus on specific panel

### **Status Bar**
- Current frame number
- Total frames
- Playback speed
- Sync status indicator
- Performance metrics (FPS)

## 🔧 Technical Considerations

### **Performance Optimization**
- **Video Decoding**: Use hardware acceleration when available
- **Memory Management**: Efficient frame buffering for smooth playback
- **Threading**: Separate threads for each video to prevent blocking
- **Caching**: Pre-load frames for seamless scrubbing

### **Synchronization Challenges**
- **Frame Rate Differences**: Handle videos with different native FPS
- **Duration Mismatches**: Automatic speed adjustment for sync
- **Seek Accuracy**: Ensure frame-perfect seeking across all videos
- **Playback Smoothness**: Maintain smooth playback despite sync calculations

### **Error Handling**
- **Missing Files**: Graceful handling of missing animations with auto-generation option
- **Codec Issues**: Support for various video formats and codecs
- **Sync Failures**: Fallback to independent playback if sync fails
- **Performance Issues**: Adaptive quality settings for smooth playback

## 📦 Dependencies

### **Core Requirements**
```txt
PyQt5>=5.15.0           # GUI framework (or PyQt6>=6.0.0)
opencv-python>=4.5.0    # Video processing
numpy>=1.20.0           # Data handling
matplotlib>=3.3.0       # For any additional plotting
```

### **Optional Enhancements**
```txt
ffmpeg-python>=0.2.0    # Advanced video processing
pillow>=8.0.0           # Image processing
pyqtgraph>=0.12.0       # High-performance plotting (alternative to matplotlib)
```

## 🎯 Success Criteria

### **Functional Requirements**
- ✅ Load and display original video
- ✅ Load and display pose animation (MP4, 50 fps)
- ✅ Load and display tracking animation (MP4, 50 fps)
- ✅ Synchronized playback across all three panels
- ✅ Unified controls (play/pause/seek/speed)
- ✅ Frame-perfect synchronization

### **Performance Requirements**
- ✅ Smooth playback at 25+ fps
- ✅ Responsive seeking and scrubbing
- ✅ Memory usage < 2GB for typical sequences
- ✅ CPU usage < 50% during playback

### **Usability Requirements**
- ✅ Intuitive interface with clear controls
- ✅ Automatic animation generation if missing
- ✅ Error messages with helpful guidance
- ✅ Keyboard shortcuts for efficient navigation

## 🚀 Quick Start Guide

### **1. Generate Required Animations**
```bash
# Generate 3D pose animation (50 fps MP4)
python scripts/visualization/animate_smpl_poses.py --sequence ARG_CRO_220001

# Generate pitch tracking animation (50 fps MP4)
python scripts/visualization/visualize_smpl_poses.py --sequence ARG_CRO_220001 --pitch-view --animate-pitch
```

### **2. Launch GUI**
```bash
# Basic usage
python gui/main.py --sequence ARG_CRO_220001

# With custom video path
python gui/main.py --video data/videos/train_data/ARG_CRO_220001.mp4 --poses results/SMPL/ARG_CRO_220001_poses_animation.mp4 --tracking results/SMPL/ARG_CRO_220001_pitch_tracking.mp4
```

### **3. Expected Workflow**
1. Select sequence from dropdown or command line
2. GUI automatically loads video and checks for animations
3. If animations missing, offers to generate them automatically
4. All three panels display synchronized content
5. Use unified controls for analysis

## 📈 Advanced Features (Future Enhancements)

### **Analysis Tools**
- Frame-by-frame comparison mode
- Pose quality metrics overlay
- Tracking accuracy indicators
- Export synchronized clips with annotations

### **Customization Options**
- Adjustable panel sizes and layouts
- Custom color schemes for pose/tracking
- Overlay options (frame numbers, timestamps)
- Multiple sequence comparison mode

### **Integration Features**
- Direct integration with visualization scripts
- Batch processing for multiple sequences
- Export to various formats (GIF, MP4, image sequences)
- Integration with evaluation pipeline

---

*This GUI will provide a comprehensive tool for analyzing the FIFA Skeletal Tracking Challenge data with perfect synchronization between video, poses, and tracking information.*