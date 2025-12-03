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

Driver fatigue is a critical factor in road traffic accidents, accounting for approximately one-fifth of annual traffic fatalities worldwide. This project implements a real-time, non-invasive driver drowsiness detection system using computer vision and deep learning techniques, optimized for deployment on the NVIDIA Jetson Nano edge computing platform.

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
Extensive review of drowsiness detection methods, deep learning approaches, and embedded system implementations.

**Survey Paper**
Modern applications and methods in drowsiness detection research.

**State-of-the-Art Analysis**
Comparison of current leading approaches and embedded system architectures.

**Implementation Reference**
Primary reference paper guiding system implementation and optimization.

**Project Logbook**
Detailed development progress, experiments, and findings throughout the project lifecycle.

**Simulation Overview**
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

### Project Members

**Mamona Sadaf**  
Research and Development Lead  
msadaf.bee22seecs@seecs.edu.pk

**Menahil Ahsan**  
Simulation and Algorithms

**Sarah Omer**  
Embedded Systems and Hardware Integration

---

## References

### Primary Publications

**Survey on Drowsiness Detection**  
Fu, B., Boutros, F., Lin, C.-T., & Damer, N.  
Comprehensive survey covering modern applications and methods in drowsiness detection across multiple domains.

**Real-Time Embedded System Implementation**  
Florez, R., Palomino-Quispe, F., Alvarez, A. B., Coaquira-Castillo, R. J., & Herrera-Levano, J. C.  
State-of-the-art embedded system achieving high accuracy with CNN and MAR analysis on Jetson Nano.

**Tiredness Detection with Jetson Nano**  
Florian, N., Popescu, D., & Hossu, A.  
Implementation reference demonstrating real-time detection using computer vision and machine learning on embedded hardware.

### Technologies

- NVIDIA Jetson Nano Platform
- OpenCV Computer Vision Library
- TensorFlow Deep Learning Framework
- MediaPipe Facial Landmark Detection
- MRL Eye Dataset

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- NVIDIA for Jetson Nano platform and development resources
- OpenCV community for computer vision tools
- Dataset contributors for training data
- NUST SEECS for research facilities and support
- Academic researchers whose work informed this project

---

## Contact

**Mamona Sadaf**  
Email: msadaf.bee22seecs@seecs.edu.pk  
GitHub: https://github.com/Mamonasadaf/Driver-Fatigue-Detection  
Institution: NUST SEECS

---

<div align="center">

**This is an ongoing research project under active development**

</div>
