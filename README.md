# FIFA Skeletal Tracking Starter Kit

This repository provides a **naïve baseline** for the **FIFA Skeletal Tracking Challenge**. It includes a simple, fully documented implementation to help participants get started with 3D pose estimation using bounding boxes, skeletal data, and camera parameters.

## 📌 Features
- **Baseline Implementation**: A simple approach for 3D skeletal tracking.
- **Camera Pose Estimation**: Computes camera transformations from bounding box correspondences.
- **Field Markings Refinement**: Improves camera rotation using detected Field Markings.
- **Pose Projection & Optimization**: Projects 3D skeletons onto 2D images and refines translation via optimization.

## 🚀 Getting Started

### 📦 Installation
Make sure you have the required dependencies installed:

```bash
pip install numpy torch opencv-python tqdm scipy
```

## 📂 Data Preparation
The script expects the following dataset structure:

```
data/
│── cameras/
│   ├── sequence1.npz
│   ├── sequence2.npz
│── images/
│   ├── sequence1/
│   │   ├── 00001.jpg
│   │   ├── 00002.jpg
│── skel_2d.npz
│── skel_3d.npz
│── boxes.npz
│── pitch_points.txt
```

- **`images/`**: Stores frame images for each sequence. **Please Ensure that the filenames are sequentially numbered** (e.g., `"00000.jpg"`, `"00001.jpg"`, etc.).
- **`cameras/`**: Contains `.npz` files with camera parameters for each sequence.
- **`boxes.npz`**: Stores bounding box data for each sequence.
- **`skel_2d.npz`**: Contains estimated 2D skeletal keypoints. 
- **`skel_3d.npz`**: Contains estimated 3D skeletal keypoints. 

You can find details about the `cameras`, `bounding boxes`, and `images` on the **Kaggle** page. For `skel_2d.npz` and `skel_3d.npz`, you can generate them automatically using the provided `preprocess.py` script. Alternatively, we have also uploaded preprocessed data [here](https://drive.google.com/drive/folders/12bu0Xmp3-euajRxIxYO92HswWWUtH-u1?usp=sharing).

### 📺 Sample Visualization
To help you visualize the results, we provide a short sample sequence in `media/sample.mp4`. 

## 🔧 Running the Baseline
To run the baseline model on the dataset, simply execute:

```bash
python baseline.py
```

By default, the script reads from the data/ directory and generates a `.npz` file (`dummy-solution.npz`) in the root folder:

You can then use the `prepare-submission.py` to create a submission file:

```bash
python prepare-submission.py -i dummy-solution.npz
```

## 📌 Notes
- This is a **naïve baseline** — you are encouraged to improve the accuracy by refining camera estimation, leveraging better keypoint tracking, or integrating deep learning approaches.

## 🤝 Contributing
If you find a bug or have suggestions for improvements, feel free to submit a pull request or open an issue.

## Acknowledgement
We use [4DHuman](https://github.com/shubham-goel/4D-Humans/tree/main) in the `preprocess.py` for estimating both 2D and 3D skeletons from bounding boxes. We appreciate the contributions of the developers and the broader research community in advancing human pose estimation.

## 📜 License
This project is licensed under the MIT License.


# Further info
from: https://www.kaggle.com/competitions/fifa-skeletal-light/data


Dataset Description
We provide camera and bounding box data for both validation (val) and test sets.

Due to .npz format limitations with nested dictionaries, camera data is stored separately per sequence, while bounding boxes are merged into a single file.

Video Access
The video footage is owned by FIFA and requires an additional agreement for access. To request permission, please complete this form. After reviewing your application, we will send you a separate license agreement along with further access details.

If you have already requested video footage from the WorldPose Dataset, you do not need to apply again, as the validation and test videos were included in that distribution.

Camera
Each camera file is stored separately per sequence in .npz format with the following structure.

{
    # Intrinsic Matrix per frame
    "K": np.array of shape (number_frames, 3, 3),  
    # Distortion coefficients per frame (k1, k2, p1, p2, k3) here only k1, k2 are valid),
    "k": np.array of shape (number_frames, 5),  
    # Rotation matrix for the first frame,
    "R": np.array of shape (1, 3, 3),  
    # Translation vector for the first frame,
    "t": np.array of shape (1, 3), 
}
To simulate a realistic setting, we provide intrinsic parameters and distortion coefficients, as modern cameras (e.g., your iPhones) often support exporting them directly. However, we only provide rotation and translation parameters for the first frame to help define the coordinate system. Participants will need to track subsequent camera poses.

Boxes
Bounding boxes are stored in a single `.npz file structured as:

{
    "<sequence_name>": np.array of shape (number_frames, Num_subjects, 4)
    # Each entry represents a bounding box per frame and subject,
    # stored in XYXY format: (x_min, y_min, x_max, y_max),
    # where (x_min, y_min) is the top-left corner
    # and (x_max, y_max) is the bottom-right corner.
    # If a subject is not present in a given frame, its bounding box is set to np.nan.
} 
Submission
For submission, keypoints should be provided in a merged file, similar to bounding boxes. Since Kaggle does not support direct submission of .npz files, we provide a conversion script to help you to convert them to the .parquet format.

{
    "<sequence_name>": np.array of shape (number_frames, Num_subjects, 15, 3), 
    # Each entry represents 3D keypoints per frame and subject,
    # stored in a (15, 3) matrix with (x, y, z) coordinates
    # for 15 selected keypoints.

    # For keypoints, we select 15 joints from **SMPL's** joint set:
    # [24, 17, 16, 19, 18, 21, 20, 2, 1, 5, 4, 8, 7, 11, 10]
    # These joints, in order, correspond to:
    # - "nose"
    # - "right_shoulder", "left_shoulder"
    # - "right_elbow", "left_elbow"
    # - "right_wrist", "left_wrist"
    # - "right_hip", "left_hip"
    # - "right_knee", "left_knee"
    # - "right_ankle", "left_ankle"
    # - "right_foot", "left_foot"

    # Please ensure you use the **SMPL** model for conversion,
    # as SMPL-H and SMPL-X have different joint orders.
}
Please ensure that your submission follows the specified format for compatibility with the evaluation system. We also provide sample submission files for your reference.