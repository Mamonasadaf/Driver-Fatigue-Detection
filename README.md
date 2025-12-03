# Real-Time Driver Drowsiness Detection System

<div align="center">

![Status](https://img.shields.io/badge/Status-In%20Development-yellow)
![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Nano-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**An embedded deep learning solution for preventing drowsy driving accidents through real-time facial analysis**

</div>

---

## Overview

Driver fatigue is a critical factor in road traffic accidents, accounting for approximately one-fifth of annual traffic fatalities worldwide. This project implements a real-time, non-invasive driver drowsiness detection system using computer vision and deep learning techniques, optimized for deployment on the NVIDIA Jetson Nano Developer Kit edge computing platform.

The system analyzes facial features captured through an infrared camera to identify drowsiness indicators including eye closure patterns, yawning frequency, and temporal behaviors. By processing these visual cues in real-time, the system provides early warning alerts to drivers before their performance deteriorates.

---

## System Pipeline

**Image Acquisition** → **Preprocessing** → **Face Detection** → **Facial Landmark Detection** → **Feature Extraction** → **CNN Classification** → **Drowsiness Logic** → **Alert System**

### Pipeline Stages

**Stage 1: Image Acquisition**
- Infrared camera captures real-time video stream
- Works in both day and night conditions
- High-resolution facial image capture

**Stage 2: Preprocessing**
- Noise reduction and image enhancement
- Grayscale conversion for efficient processing
- Normalization for consistent input

**Stage 3: Face Detection**
- Haar Cascade classifier locates face in frame
- Bounding box extraction around face region
- Handles various head orientations

**Stage 4: Facial Landmark Detection**
- MediaPipe Face Mesh detects precise facial points
- Identifies eye and mouth regions
- Compensates for head movement and rotation

**Stage 5: Feature Extraction**
- Extract eye regions for analysis
- Calculate Eye Aspect Ratio
- Calculate Mouth Aspect Ratio
- ROI stabilization for accurate measurements

**Stage 6: CNN Classification**
- Custom lightweight neural network
- Classifies eye state as open or closed
- Optimized for embedded hardware performance

**Stage 7: Drowsiness Logic**
- Score-based state tracking
- Temporal filtering to distinguish blinks from drowsiness
- Yawn frequency monitoring
- PERCLOS analysis for sustained eye closure

**Stage 8: Alert System**
- Visual warnings on display
- Audio alerts through GPIO buzzer
- Event logging with timestamps

---

## Key Features

### Detection Capabilities
- Real-time eye closure monitoring
- Yawn detection and frequency tracking
- Multi-modal analysis combining multiple indicators
- Temporal filtering to reduce false alarms
- Adaptive threshold configuration

### Technical Highlights
- Custom lightweight CNN architecture
- MediaPipe facial landmark detection
- Haar Cascade face localization
- CUDA acceleration for GPU inference
- Eye Aspect Ratio geometric analysis
- Mouth Aspect Ratio yawn detection
- Day and night operation support

---

## Hardware Requirements

### Essential Components
- NVIDIA Jetson Nano Developer Kit
- Raspberry Pi Camera Module or Infrared Camera
- MicroSD Card for storage
- Power supply for Jetson Nano
- GPIO buzzer for audio alerts

### Platform Specifications
- GPU acceleration for deep learning inference
- Quad-core ARM processor
- Sufficient memory for model deployment
- Camera interface support
- GPIO pins for peripherals

---

## Software Requirements

### Core Technologies
- Ubuntu operating system
- JetPack SDK with CUDA support
- Python programming environment
- OpenCV for computer vision
- TensorFlow and Keras for deep learning
- MediaPipe for facial landmark detection

### Development Tools
- NumPy for numerical computations
- SciPy for scientific computing
- Matplotlib for visualization
- Additional utilities for image processing

---

## Installation

### Initial Setup
- Flash JetPack SDK to microSD card
- Boot Jetson Nano and complete system configuration
- Update system packages and dependencies

### System Dependencies
- Install Python development tools
- Install OpenCV and image processing libraries
- Install deep learning frameworks
- Install camera interface utilities
- Install GPIO control libraries

### Project Setup
- Clone repository from GitHub
- Install Python package requirements
- Download pre-trained models and classifiers
- Configure camera interface
- Set up GPIO pins for buzzer

### Verification
- Test camera functionality
- Verify model loading
- Check GPU acceleration
- Confirm all dependencies installed

---

## Usage

### Running the System
The system can be launched with various configuration options to suit different deployment scenarios.

### Basic Operation
- Start drowsiness detection with default settings
- Enable visual display for monitoring
- Activate audio alerts when drowsiness detected
- Log detection events for analysis

### Configuration Options
- Adjust detection sensitivity thresholds
- Modify alert trigger parameters
- Configure camera resolution and frame rate
- Enable or disable specific features
- Customize alert mechanisms

### Training Custom Models
- Prepare dataset with labeled images
- Train CNN model on eye state classification
- Evaluate model performance on test data
- Export trained model for deployment

### Production Deployment
- Run as background system service
- Enable automatic startup on boot
- Configure continuous logging
- Set up remote monitoring capabilities

---

## Technical Implementation

### Algorithm Components

**MediaPipe Face Mesh**
- Detects precise facial landmarks in real-time
- Robust to head rotation and lighting variations
- Provides accurate eye and mouth localization
- Handles partial occlusions effectively

**ROI Correction Algorithm**
- Stabilizes eye region during head movement
- Compensates for lateral and vertical inclination
- Prevents information loss during tracking
- Maintains consistent region dimensions

**CNN Eye Classification**
- Custom lightweight architecture for embedded deployment
- Processes grayscale eye region images
- Multiple convolutional layers for feature learning
- Dropout regularization to prevent overfitting
- Binary classification output for eye state

**Eye Aspect Ratio**
- Geometric metric quantifying eye openness
- Based on relationship between vertical and horizontal landmarks
- Threshold-based detection of eye closure
- Sustained closure indicates drowsiness

**Mouth Aspect Ratio**
- Measures mouth opening to detect yawning
- Geometric calculation using mouth landmarks
- Frequency tracking over time windows
- Multiple yawns indicate fatigue

**Temporal Logic**
- Score-based tracking of drowsiness state
- Distinguishes between normal blinks and microsleep
- Sliding window analysis for PERCLOS
- Hysteresis to prevent alert oscillation

### Optimization Strategies

**Model Compression**
- Reduced parameter count for faster inference
- Quantization for lower precision computation
- Pruning to remove redundant connections

**Hardware Acceleration**
- CUDA kernel optimization for GPU
- Efficient memory management
- Asynchronous frame processing
- Thermal management for sustained performance

---

## Performance Characteristics

### Model Accuracy
The custom CNN model demonstrates strong performance across training, validation, and test datasets with high accuracy in eye state classification.

### Hardware Performance
The system achieves real-time processing rates suitable for automotive applications with acceptable inference latency and resource utilization on the Jetson Nano platform.

### Detection Performance
The system maintains reliable detection across various conditions including different lighting environments, head orientations, and partial occlusions.

### Efficiency Comparison
The custom lightweight CNN provides optimal balance between accuracy and inference speed compared to larger pre-trained models, making it suitable for embedded deployment.

---

## Project Structure

### Documentation
- Comprehensive literature review
- Implementation reference papers
- State-of-the-art comparison analysis
- Project development logbook
- System overview presentations

### Models
- Pre-trained CNN model files
- Haar Cascade classifiers
- Model architecture definitions

### Source Code
- Main drowsiness detection script
- Model training implementation
- Preprocessing utilities
- Feature extraction modules
- Face and eye detection components
- Yawn detection logic
- Alert system implementation
- Camera interface utilities
- Configuration management

### Data
- Training dataset organization
- Validation set structure
- Test data arrangement
- Dataset preprocessing scripts

### Development Resources
- Jupyter notebooks for experimentation
- Model training workflows
- Performance analysis tools
- Visualization utilities

### Testing
- Unit tests for components
- Integration testing
- Performance benchmarking

### Utilities
- Dataset preparation scripts
- Installation verification tools
- Data augmentation utilities
- Model conversion helpers

---

## Dataset

### Current Dataset: MRL Eye Dataset

The project currently utilizes the MRL Eye Dataset for training and evaluating the CNN-based drowsiness detection model. This dataset provides diverse samples of eye states under various conditions.

**Dataset Characteristics**
- Labeled eye images for supervised learning
- Multiple subjects with demographic diversity
- Various lighting conditions represented
- Open and closed eye state categories

**Data Organization**
- Training subset for model learning
- Validation subset for hyperparameter tuning
- Test subset for final evaluation
- Balanced class distribution

**Preprocessing Pipeline**
- Image resizing for uniform input dimensions
- Grayscale conversion for computational efficiency
- Pixel value normalization
- Quality filtering to remove poor samples

**Data Augmentation**
- Rotation to handle head tilt
- Horizontal flipping for left-right invariance
- Brightness adjustment for lighting robustness
- Minor zoom and shift transformations
- Noise injection for improved generalization

### Additional Reference Datasets

**NTHU-DDD**
National Tsing Hua University Drowsy Driver Detection dataset providing comprehensive drowsiness scenarios.

**YawDD**
Yawning Detection Dataset with video sequences for temporal analysis.

**UTA-RLDD**
University of Texas at Arlington Real-Life Drowsiness Dataset captured in authentic driving conditions.

**NITYMED**
Night-Time Yawning-Microsleep-Eyeblink-Driver Distraction dataset emphasizing low-light performance.

---

## Documentation

### Project Documents

Comprehensive documentation is available in the repository:

**Literature Review**  
[View Document](https://github.com/Mamonasadaf/Driver-Fatigue-Detection/blob/main/doc/Literature%20review.%20%20(1).pdf)  
Extensive review of drowsiness detection methods, deep learning approaches, and embedded system implementations.

**Survey Paper**  
[View Document](https://github.com/Mamonasadaf/Driver-Fatigue-Detection/blob/main/doc/3%20main%20papers/survey.paper.pdf)  
Modern applications and methods in drowsiness detection research.

**State-of-the-Art Analysis**  
[View Document](https://github.com/Mamonasadaf/Driver-Fatigue-Detection/blob/main/doc/3%20main%20papers/State%20of%20art.pdf)  
Comparison of current leading approaches and embedded system architectures.

**Implementation Reference Paper**  
[View Document](https://github.com/Mamonasadaf/Driver-Fatigue-Detection/blob/main/doc/3%20main%20papers/main_paper1.pdf)  
Primary reference paper guiding system implementation and optimization.

**Project Logbook**  
[View Document](https://github.com/Mamonasadaf/Driver-Fatigue-Detection/blob/main/doc/logbook.md)  
Detailed development progress, experiments, and findings throughout the project lifecycle.

**Simulation Overview**  
[View Document](https://github.com/Mamonasadaf/Driver-Fatigue-Detection/blob/main/doc/step%201_Beginner%20overview%20of%20simulation.pptx)  
Beginner-friendly guide to system simulation and testing procedures.

### System Configuration

Detection thresholds, camera settings, model paths, and alert parameters can be customized through configuration files to adapt the system to specific requirements and deployment scenarios.

---

## Current Development Status

This project is actively under development. Current focus areas include:

### Completed Components
- System architecture design
- Literature review and research
- Hardware platform selection
- Basic pipeline implementation
- Initial model training with MRL dataset

### In Progress
- CNN model optimization for Jetson Nano
- Real-time performance tuning
- Alert system integration
- Comprehensive testing across conditions
- Dataset expansion and augmentation

### Planned Development
- Advanced temporal analysis
- Multi-modal sensor fusion
- Enhanced robustness to occlusions
- Production deployment optimization
- Field testing and validation

---

## Future Enhancements

### Near-Term Goals
- Model optimization with TensorRT
- Improved inference speed
- Enhanced detection accuracy
- Robust alert mechanisms
- Comprehensive logging system

### Long-Term Vision
- Integration with vehicle systems
- Cloud connectivity for fleet monitoring
- Mobile application development
- Advanced analytics and reporting
- Regulatory compliance certification

### Research Directions
- Temporal pattern recognition with recurrent networks
- Attention mechanisms for improved accuracy
- Multi-modal fusion with physiological sensors
- Federated learning for privacy preservation
- Cross-dataset generalization studies

---

## Team

**National University of Sciences and Technology (NUST)**  
School of Electrical Engineering and Computer Science (SEECS)  
Department of Electrical Engineering  
Islamabad, Pakistan

### Supervisor

**Dr. Tauseef Ur Rehman**  
Project Supervisor  
GitHub: [Tauseef-dr](https://github.com/Tauseef-dr)

### Project Members

**Mamona Sadaf**  
Research and Development Lead  
Email: msadaf.bee22seecs@seecs.edu.pk

**Menahil Ahsan**  
Simulation and Algorithms

**Sarah Omer**  
Embedded Systems and Hardware Integration

### Teaching Assistant

**Zahid**  
Teaching Assistant

### Lab Support

**Miss Tehniyyat Siddique**  
Lab Engineer

---

## References

### Primary Publications

**A Survey on Drowsiness Detection – Modern Applications and Methods**  
Fu, B., Boutros, F., Lin, C.-T., & Damer, N. (2024)  
arXiv preprint arXiv:2408.12990  
[https://arxiv.org/abs/2408.12990](https://arxiv.org/abs/2408.12990)  
Comprehensive survey covering modern applications and methods in drowsiness detection across multiple domains.

**A Real-Time Embedded System for Driver Drowsiness Detection**  
Florez, R., Palomino-Quispe, F., Alvarez, A. B., Coaquira-Castillo, R. J., & Herrera-Levano, J. C. (2024)  
Sensors, 24(19), 6261  
[https://www.mdpi.com/1424-8220/24/19/6261](https://www.mdpi.com/1424-8220/24/19/6261)  
State-of-the-art embedded system achieving high accuracy with CNN and MAR analysis on Jetson Nano.

**Real-Time Tiredness Detection System Using Nvidia Jetson Nano and OpenCV**  
Florian, N., Popescu, D., & Hossu, A. (2024)  
Procedia Computer Science, 242, 536-543  
[https://www.sciencedirect.com/science/article/pii/S1877050924018209](https://www.sciencedirect.com/science/article/pii/S1877050924018209)  
Implementation reference demonstrating real-time detection using computer vision and machine learning on embedded hardware.

### Technologies and Platforms

**NVIDIA Jetson Nano Developer Kit**  
[https://developer.nvidia.com/embedded/jetson-nano-developer-kit](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)  
Edge AI computing platform used for embedded deployment.

**OpenCV Computer Vision Library**  
[https://opencv.org/](https://opencv.org/)  
Open-source computer vision and machine learning software library.

**TensorFlow Deep Learning Framework**  
[https://www.tensorflow.org/](https://www.tensorflow.org/)  
End-to-end open-source platform for machine learning.

**MediaPipe Facial Landmark Detection**  
[https://mediapipe.dev/](https://mediapipe.dev/)  
Cross-platform framework for building multimodal applied ML pipelines.

**MRL Eye Dataset**  
Dataset used for CNN-based drowsiness detection model training.

---

## Acknowledgments

### Special Thanks

We extend our sincere gratitude to:

**Dr. Tauseef Ur Rehman**  
Our project supervisor for guidance, support, and expertise throughout this research project.  
GitHub: [Tauseef-dr](https://github.com/Tauseef-dr)

**Jetson Nano Warriors Community**  
For their valuable resources, tutorials, and support in learning and implementing solutions on the Jetson Nano platform.

**Zahid (Teaching Assistant)**  
For technical assistance and guidance during development and testing phases.

**Miss Tehniyyat Siddique (Lab Engineer)**  
For laboratory facilities support and hardware troubleshooting assistance.

### Additional Acknowledgments

- NVIDIA for the Jetson Nano Developer Kit and comprehensive development resources
- OpenCV community for robust computer vision tools and documentation
- TensorFlow and MediaPipe teams for powerful machine learning frameworks
- Dataset contributors for providing training and validation data
- NUST SEECS for research facilities, infrastructure, and institutional support
- All researchers and academic contributors whose work informed this project

---

## License

This project is licensed under the MIT License.

---

## Contact

**Primary Contact**  
Mamona Sadaf  
Email: msadaf.bee22seecs@seecs.edu.pk

**Project Repository**  
GitHub: [https://github.com/Mamonasadaf/Driver-Fatigue-Detection](https://github.com/Mamonasadaf/Driver-Fatigue-Detection)

**Institution**  
National University of Sciences and Technology (NUST)  
School of Electrical Engineering and Computer Science (SEECS)  
Islamabad, Pakistan

---

<div align="center">

**This is an ongoing research project under active development**

**Powered by NVIDIA Jetson Nano Developer Kit**

</div>
