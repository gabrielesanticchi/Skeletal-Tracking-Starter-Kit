# SMPL Poses Visualization Examples

This directory contains example outputs from the SMPL pose visualization tools for the FIFA Skeletal Tracking Challenge training data.

## 📁 Generated Examples

### Original Examples
### 1. **example_3d_poses.png** - Static 3D Pose Visualization (Basic)
- **Sequence**: ARG_CRO_220001, **Frame**: 100, **Subjects**: 5 (with labels)
- **Description**: Shows 3D skeletal poses in world coordinates with simplified joint positioning

### 2. **example_pitch_tracking.png** - Static Pitch Tracking Visualization
- **Sequence**: ARG_CRO_220001, **Frame Range**: 0-200 (step: 10), **Subjects**: 5
- **Description**: Top-down view showing player trajectories on the football pitch with start/end markers

### 3. **example_animation.gif** - Basic 3D Animation (GIF)
- **Sequence**: ARG_CRO_220001, **Frame Range**: 50-60 (step: 2), **Subjects**: 3, **FPS**: 10
- **Description**: Basic animated 3D skeletal movement (translation-based)

### 4. **example_animation.mp4** - Basic 3D Animation (MP4)
- **Sequence**: ARG_CRO_220001, **Frame Range**: 100-120 (step: 2), **Subjects**: 3, **FPS**: 15
- **Description**: High-quality MP4 animation (translation-based)

### Improved Examples (With Actual Joint Animation)
### 5. **improved_3d_poses.png** - Enhanced 3D Pose Visualization
- **Sequence**: ARG_CRO_220001, **Frame**: 150, **Subjects**: 3 (with labels)
- **Description**: Shows 3D poses using actual SMPL pose parameters with proper joint rotations and forward kinematics

### 6. **improved_joint_animation.gif** - True Joint Animation
- **Sequence**: ARG_CRO_220001, **Frame Range**: 50-70 (step: 2), **Subjects**: 2, **FPS**: 10
- **Description**: Animated 3D skeletons with actual joint movements based on SMPL pose parameters

### 7. **animated_pitch_tracking.gif** - Animated Pitch Tracking
- **Sequence**: ARG_CRO_220001, **Frame Range**: 0-100 (step: 5), **Subjects**: 3, **FPS**: 15
- **Description**: Real-time animated pitch tracking with movement trails showing player trajectories over time

## 🎯 Key Features Demonstrated

### Enhanced 3D World Coordinate Visualization
- ✅ Poses shown in original SMPL world coordinate system
- ✅ **NEW**: Actual joint animation using SMPL pose parameters (global_orient, body_pose)
- ✅ **NEW**: Forward kinematics with proper joint rotations
- ✅ **NEW**: Rodrigues rotation formula for axis-angle conversions
- ✅ Anatomically correct skeleton structure (24 joints)
- ✅ Color-coded subjects for easy identification
- ✅ Optional subject labels and customizable view angles

### Advanced Pitch Tracking Capabilities
- ✅ X,Y coordinate extraction from SMPL translation vectors
- ✅ **NEW**: Animated pitch tracking with movement trails
- ✅ **NEW**: Real-time trajectory visualization
- ✅ **NEW**: Configurable trail length for movement history
- ✅ Static trajectory visualization with start/end markers
- ✅ Perfect for studying player movement and positioning patterns

### Comprehensive Animation Support
- ✅ **NEW**: True joint animation (not just translation)
- ✅ **NEW**: Animated pitch tracking with trails
- ✅ Smooth frame-by-frame animation
- ✅ Multiple export formats (GIF, MP4)
- ✅ Configurable FPS for video synchronization
- ✅ Frame stepping for performance optimization

### Robust Data Handling
- ✅ **NEW**: Enhanced NaN/Inf value handling in pose calculations
- ✅ **NEW**: Fallback to neutral poses for corrupted data
- ✅ Subject filtering and selection
- ✅ Frame range selection with validation
- ✅ Automatic axis scaling and centering

## 🚀 Usage Examples

### Static 3D Visualization
```bash
# Basic 3D pose visualization
python scripts/visualization/visualize_smpl_poses.py --sequence ARG_CRO_220001 --frame 100

# With labels and subject filtering
python scripts/visualization/visualize_smpl_poses.py --sequence ARG_CRO_220001 --frame 100 --show-labels --num-subjects 5 --output results/SMPL/my_3d_poses.png
```

### Pitch Tracking
```bash
# Trajectory visualization
python scripts/visualization/visualize_smpl_poses.py --sequence ARG_CRO_220001 --pitch-view --start-frame 0 --end-frame 200 --frame-step 10 --num-subjects 5 --output results/SMPL/my_tracking.png
```

### Animation Creation
```bash
# 3D Joint Animation (GIF)
python scripts/visualization/animate_smpl_poses.py --sequence ARG_CRO_220001 --start-frame 50 --end-frame 100 --fps 10 --output results/SMPL/my_animation.gif

# 3D Joint Animation (MP4 for video sync)
python scripts/visualization/animate_smpl_poses.py --sequence ARG_CRO_220001 --start-frame 100 --end-frame 200 --fps 25 --output results/SMPL/my_video.mp4

# Animated Pitch Tracking
python scripts/visualization/visualize_smpl_poses.py --sequence ARG_CRO_220001 --pitch-view --animate-pitch --start-frame 0 --end-frame 200 --fps 20 --trail-length 40 --output results/SMPL/my_tracking.gif
```

## 📊 Technical Details

### Enhanced SMPL Joint Structure (24 joints)
The visualization now uses proper SMPL forward kinematics with:
- **Core**: Pelvis (root), Spine1-3, Neck, Head
- **Arms**: Left/Right Collar, Shoulder, Elbow, Wrist, Hand
- **Legs**: Left/Right Hip, Knee, Ankle, Foot
- **NEW**: Proper joint hierarchy and parent-child relationships
- **NEW**: Rodrigues rotation formula for axis-angle to rotation matrix conversion
- **NEW**: Forward kinematics for accurate joint positioning

### Data Sources & Processing
- **Training Sequences**: 89 sequences with SMPL poses available
- **World Coordinates**: Uses SMPL translation vectors for positioning
- **Pose Parameters**: Utilizes global_orient, body_pose, transl, and betas
- **NEW**: Real-time pose parameter processing for joint animation
- **NEW**: Robust handling of corrupted or missing pose data

### Animation Capabilities
- **Joint Animation**: True skeletal animation using SMPL pose parameters
- **Pitch Tracking**: Animated movement trails with configurable history
- **Export Formats**: GIF (preview), MP4 (high-quality)
- **Performance**: Frame stepping and subject limiting for optimization

### Performance Considerations
- **Frame Stepping**: Use `--frame-step` to reduce processing time
- **Subject Limiting**: Use `--num-subjects` to focus on specific players
- **Format Choice**: GIF for previews, MP4 for high-quality output

## 🔧 Troubleshooting

### Common Issues
1. **NaN/Inf Values**: Automatically handled by replacing with origin coordinates
2. **Missing Sequences**: Only training data (89 sequences) have SMPL poses
3. **Animation Performance**: Use frame stepping and subject limiting for faster processing
4. **MP4 Export**: Requires ffmpeg installation for video output

### Data Validation
The visualization tools automatically:
- Filter out invalid coordinate values
- Handle missing or corrupted pose data
- Provide fallback axis limits for empty datasets
- Validate frame ranges and subject indices

## 📈 Analysis Applications

### Training Data Analysis
- Visualize pose quality and consistency across sequences
- Identify problematic frames or subjects
- Study player movement patterns and positioning

### Algorithm Development
- Validate pose estimation algorithms
- Compare predicted vs. ground truth poses
- Develop tracking and prediction models

### Presentation and Documentation
- Create compelling visualizations for papers/presentations
- Generate synchronized videos for analysis
- Demonstrate dataset capabilities and quality

---

*Generated using the FIFA Skeletal Tracking Challenge SMPL visualization tools*