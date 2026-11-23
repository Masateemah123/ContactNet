# ContactNet

### Progressive CNN-GRU for Proprioceptive Contact Estimation in Quadrupedal Robots

ContactNet is a deep learning framework for estimating foot-contact states in quadrupedal robots using proprioceptive measurements.

Developed as part of graduate research at the **University of São Paulo (USP), Brazil**, ContactNet extends an existing CNN-based contact estimation framework to the **Unitree Go1 quadruped** through recurrent temporal modelling, robot-specific data acquisition and processing, and independent model training and evaluation.

The framework introduces a Progressive CNN-GRU architecture that combines convolutional feature extraction with recurrent temporal modelling at multiple feature extraction stages. It estimates the contact configuration of the robot's four legs using proprioceptive measurements, without requiring IMU measurements as network inputs.

---

## Overview

Foot-contact estimation is an important component of legged robot state estimation and locomotion. Reliable identification of contact states provides information about the interaction between a robot and its environment, supporting state estimation and contact-aware locomotion.

ContactNet investigates learning-based contact estimation using temporal sequences of proprioceptive measurements collected from a Unitree Go1 quadruped.

Unlike the original CNN-based implementation on which this work builds, ContactNet introduces recurrent GRU branches to capture temporal dependencies at different levels of convolutional feature extraction.

The network predicts one of 16 possible contact configurations, representing the binary contact states of the four robot legs.

### Key Features

- Adaptation of an existing CNN-based contact estimation framework to the Unitree Go1.
- Progressive CNN-GRU architecture with early and late recurrent branches.
- Multi-scale temporal feature fusion.
- Proprioceptive-only input representation with 48 features.
- Temporal sequence processing using 150-sample windows.
- ROS bag-based robot data acquisition and processing.
- Four-leg contact configuration classification.
- Individual-leg contact evaluation.

---

## Research Contributions

ContactNet extends the original Deep Contact Estimator framework through the following research and engineering contributions.

### 1. Cross-platform adaptation

Adaptation of the original CNN-based contact estimation framework, developed for the MIT Cheetah platform, to the Unitree Go1 quadruped.

This required adapting the input representation, robot-specific proprioceptive measurements, leg ordering conventions, contact labels, and data-processing pipeline.

### 2. Progressive CNN-GRU architecture

Extension of the CNN-based feature extraction architecture through the integration of two GRU branches operating at different feature extraction stages.

The early GRU processes intermediate CNN features, while the late GRU processes higher-level CNN representations. Their outputs are combined through multi-scale temporal feature fusion before classification.

### 3. Go1 proprioceptive data acquisition

Collection of robot-state measurements from Unitree Go1 ROS bag recordings, providing the data foundation for training and evaluating the adapted contact estimation framework.

### 4. Robot-specific data processing pipeline

Development of a processing workflow for extracting and organizing joint states, end-effector kinematics, robot base pose, and contact labels from recorded ROS data.

The pipeline includes coordinate-frame transformations, feature organization, leg reordering, contact-label encoding, and dataset generation.

### 5. IMU-independent proprioceptive representation

Development and evaluation of a 48-dimensional input representation comprising joint positions, joint velocities, body-frame foot positions, and body-frame foot velocities.

IMU measurements are not included in the final network input representation.

### 6. Model training and evaluation

Implementation of the training and evaluation pipeline using PyTorch and PyTorch Lightning, with multi-class contact configuration prediction and individual-leg contact performance assessment.

---

## Model Architecture

ContactNet uses a **Progressive CNN-GRU architecture** that integrates convolutional feature extraction and recurrent temporal modelling.

The architecture consists of three convolutional blocks and two GRU branches.

The first GRU operates on intermediate representations extracted after the second convolutional block. The second GRU operates on higher-level representations extracted after the third convolutional block.

The final temporal representations are concatenated and passed to a fully connected classification head.

### Architecture Configuration

| Component | Configuration |
|---|---|
| Input shape | 150 × 48 |
| Input channels | 48 |
| Temporal window | 150 samples |
| Convolutional blocks | 3 |
| CNN channel progression | 48 → 64 → 128 → 256 |
| Early GRU hidden size | 64 |
| Late GRU hidden size | 64 |
| Fused representation | 128 features |
| Classification head | 128 → 512 → 256 → 16 |
| Output classes | 16 |

### Network Structure

```text
Input Sequence
(B, 150, 48)
       |
       v
   CNN Block 1
   48 → 64
       |
       v
   CNN Block 2
   64 → 128
       |
       +----------------------+
       |                      |
       v                      |
   Early GRU                  |
   128 → 64                   |
       |                      |
       v                      |
  Early Features              |
       |                      |
       |                 CNN Block 3
       |                 128 → 256
       |                      |
       |                      v
       |                   Late GRU
       |                   256 → 64
       |                      |
       |                      v
       |                 Late Features
       |                      |
       +----------+-----------+
                  |
                  v
         Feature Concatenation
             128 Features
                  |
                  v
            Classifier
          128 → 512 → 256
                  |
                  v
             16 Classes
```

### Input Representation

The network receives four groups of proprioceptive measurements:

| Input Feature | Dimensions | Description |
|---|---:|---|
| Joint positions | 12 | Positions of the 12 actuated joints |
| Joint velocities | 12 | Velocities of the 12 actuated joints |
| Foot positions | 12 | Body-frame positions of the four feet |
| Foot velocities | 12 | Body-frame velocities of the four feet |
| **Total** | **48** | **Proprioceptive input features** |

Each temporal input window contains 150 consecutive samples of these 48 features.

### Output Representation

The network performs 16-class classification.

Each class represents one possible binary contact configuration of the four robot legs:

- Four binary contact states.
- Two possible states per leg: contact or no contact.
- Sixteen possible four-leg configurations.

The output logits represent the predicted contact configuration.

---

## Data Acquisition and Processing

The ContactNet dataset was developed using proprioceptive measurements collected from Unitree Go1 ROS bag recordings.

The processing workflow converts recorded robot-state measurements into structured temporal datasets suitable for supervised learning.

### 1. ROS Bag Extraction

ROS bag recordings are processed using `bagpy` to extract robot-state measurements.

The processing pipeline supports compressed `.zst` recordings through `zstandard`.

The following ROS topics are extracted:

| ROS Topic | Purpose |
|---|---|
| `/estimated_robot_state` | Robot joint states, end-effector measurements, base pose, and contact information |
| `/gazebo/imu` | Recorded IMU measurements, processed separately from the final model inputs |

Extracted measurements are initially exported into CSV files and subsequently organized into Parquet datasets.

### 2. Proprioceptive Signal Extraction

The following measurements are extracted from the robot-state recordings:

- Joint positions: `joint_state.position`
- Joint velocities: `joint_state.velocity`
- End-effector positions: `ee_pos`
- End-effector velocities: `ee_vel`
- End-effector contact information: `ee_contact`
- Base position: `base_pose.position`
- Base orientation: `base_pose.orientation`

The four legs are represented using their corresponding end-effector measurements.

### 3. Coordinate-Frame Transformation

End-effector positions and velocities are transformed from the world frame into the robot body frame.

The transformation uses the robot's base position and orientation.

Quaternion normalization and rotation matrices are used to perform the coordinate transformations.

This produces body-frame foot positions and velocities for the four legs.

### 4. Feature Organization

The extracted signals are organized into four proprioceptive feature groups:

- Joint positions.
- Joint velocities.
- Body-frame foot positions.
- Body-frame foot velocities.

The measurements are reordered according to the selected leg convention to maintain consistency between the input features and contact labels.

### 5. Contact Label Generation

The contact information extracted from `ee_contact` provides the reference contact states.

The four binary leg-contact states are reordered according to the model's leg convention and encoded into 16 categorical contact configurations.

### 6. Dataset Generation

The processed measurements are saved as NumPy arrays for subsequent training and evaluation.

| File | Description |
|---|---|
| `q.npy` | Joint positions |
| `qd.npy` | Joint velocities |
| `p.npy` | Body-frame foot positions |
| `v.npy` | Body-frame foot velocities |
| `contacts.npy` | Four-leg binary contact states |
| `contacts_labels.npy` | Encoded 16-class contact labels |
| `preprocessed_data.npy` | Concatenated 48-dimensional proprioceptive features |

The resulting feature representation is used to construct temporal input sequences for the ContactNet training pipeline.

---

## Training Framework

The ContactNet training implementation uses:

- Python
- PyTorch
- PyTorch Lightning
- NumPy
- SciPy

### Training Configuration

The implementation supports:

- Multi-class focal loss.
- Adam optimization.
- Weight decay.
- Gradient clipping.
- Validation monitoring.
- Model checkpoint selection.
- Configurable learning-rate scheduling.

The focal loss implementation uses a configurable focusing parameter, with a default gamma of 2.0.

### Evaluation Metrics

Performance is evaluated at both the contact-configuration level and the individual-leg level.

Evaluation metrics include:

- Multi-class contact configuration accuracy.
- Individual-leg contact accuracy.
- Per-leg precision and recall.
- Per-leg F1-score.
- Confusion matrices.
- Classification reports.

These metrics provide complementary information about the network's ability to identify complete contact configurations and estimate individual foot-contact states.

---

## Research Publications

### Conference Publication

**Comprehensive Review of Contact Estimation Methods for Legged Robots**

Authors: Thema Masa, Vivian Suzano Medeiros, Marcelo Becker.

2025 Brazilian Conference on Robotics (CROS).

[View publication on IEEE Xplore](https://ieeexplore.ieee.org/document/11066135/)

### Master's Research

**Master's Research — University of São Paulo (USP), Brazil**

[View research record in the USP Digital Repository](https://repositorio.usp.br/item/003319254)

---

## Original Framework and Acknowledgements

ContactNet builds upon the open-source [Deep Contact Estimator](https://github.com/UMich-CURLY/deep-contact-estimator), developed by Tzu-Yuan Lin, Ray Zhang, Justin Yu, and Maani Ghaffari at the University of Michigan.

The original implementation provided the CNN-based contact estimation foundation used as a starting point for this research. Its original work focused on the MIT Cheetah platform.

ContactNet represents a subsequent adaptation and extension developed for the Unitree Go1. The present research includes the integration of recurrent temporal modelling, Go1-specific data acquisition and processing, a proprioceptive input representation, and independent model training and evaluation using Go1 recordings.

The original implementation and its associated research are acknowledged as the foundation of this work. The ContactNet dataset and subsequent Go1-specific research contributions are distinct from the original UMich-CURLY dataset and implementation.

### Original Publication

Tzu-Yuan Lin, Ray Zhang, Justin Yu, and Maani Ghaffari.

*Legged Robot State Estimation using Invariant Kalman Filtering and Learned Contact Events.*

Proceedings of the 5th Conference on Robot Learning (CoRL), 2021.

- [Original paper — OpenReview](https://openreview.net/forum?id=yt3tDB67lc5)
- [Original source code — GitHub](https://github.com/UMich-CURLY/deep-contact-estimator)

### Original Citation

```bibtex
@inproceedings{lin2021legged,
  title={Legged Robot State Estimation using Invariant Kalman Filtering and Learned Contact Events},
  author={Lin, Tzu-Yuan and Zhang, Ray and Yu, Justin and Ghaffari, Maani},
  booktitle={5th Annual Conference on Robot Learning},
  year={2021},
  url={https://openreview.net/forum?id=yt3tDB67lc5}
}
```

---

## Technologies

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch, PyTorch Lightning |
| Data Processing | NumPy, Pandas, SciPy |
| Robotics Middleware | ROS |
| Data Extraction | bagpy, zstandard |
| Data Storage | CSV, Parquet, NumPy |
| Neural Architectures | CNN, GRU, CNN-GRU |
| Research Domain | Quadrupedal Robotics, Proprioceptive Contact Estimation |

---

## Author

**Masa Thema**

Mechanical Engineering  
University of São Paulo (USP)  
São Carlos, Brazil

---

*ContactNet — Progressive Temporal Learning for Proprioceptive Contact Estimation in Quadrupedal Robots.*
