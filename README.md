# Real-Time Driver Drowsiness Detection System

![Status](https://img.shields.io/badge/Status-In%20Development-yellow)
![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Nano-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)

**An embedded deep learning solution for preventing drowsy driving accidents through real-time facial analysis**

---

## Overview

Driver fatigue is a critical factor in road traffic accidents. This project implements a real-time driver drowsiness detection system using computer vision and deep learning, optimized for deployment on the NVIDIA Jetson Nano edge computing platform.

The system analyzes facial features captured through a camera to identify drowsiness indicators including eye closure patterns and yawning frequency.

---

## Planned System Pipeline

**Image Acquisition** → **Face Detection** → **Facial Landmark Detection** → **Feature Extraction** → **CNN Classification** → **Drowsiness Logic** → **Alert System**

---

## Key Features (Planned)

- Real-time eye closure monitoring
- Yawn detection
- MediaPipe facial landmark detection
- Custom CNN for eye state classification
- Alert system with visual and audio warnings

---

## Hardware

- NVIDIA Jetson Nano Developer Kit
- Camera Module (Raspberry Pi Camera or USB Camera)
- MicroSD Card
- Power supply
- GPIO buzzer for alerts

---

## Software Requirements

- Ubuntu (via JetPack SDK)
- Python 3.8+
- OpenCV
- TensorFlow/Keras
- MediaPipe
- NumPy

---

## Current Development Status

### Completed
- Literature review on drowsiness detection methods
- Research paper selection and analysis
- GitHub repository setup
- Jetson Nano flashing and configuration
- USB boot setup
- Essential software installations (OpenCV, Python libraries)
- Initial testing with laptop camera

### In Progress
- Algorithm implementation
- CNN model preparation
- Camera integration with Jetson Nano
- Detection pipeline development

### Planned
- CNN model training with MRL Eye Dataset
- Real-time deployment on Jetson Nano
- Alert system integration
- Performance optimization
- Comprehensive testing

---

## Project Structure

```
Driver-Fatigue-Detection/
├── doc/                          # Documentation
│   ├── Literature review.pdf     # Comprehensive research review
│   ├── 3 main papers/            # Reference papers
│   └── logbook.md                # Development progress log
├── models/                       # Model files (to be added)
├── src/                          # Source code (in development)
└── README.md                     # This file
```

---

## Documentation

**Literature Review**  
[View Document](https://github.com/Mamonasadaf/Driver-Fatigue-Detection/blob/main/doc/Literature%20review.%20%20(1).pdf)

**Project Logbook**  
[View Document](https://github.com/Mamonasadaf/Driver-Fatigue-Detection/blob/main/doc/logbook.md)

---

## Dataset

**MRL Eye Dataset**  
Used for training the CNN-based eye state classification model.

---

## Team

**National University of Sciences and Technology (NUST)**  
School of Electrical Engineering and Computer Science (SEECS)

### Supervisor
**Dr. Tauseef Ur Rehman**  
GitHub: [Tauseef-dr](https://github.com/Tauseef-dr)

### Team Members
- **Mamona Sadaf** - Research & Development Lead
- **Menahil Ahsan** - Algorithms & Simulation
- **Sarah Omer** - Embedded Systems

### Support
- **Zahid** - Teaching Assistant
- **Miss Tehniyyat Siddique** - Lab Engineer

---

## References

**A Survey on Drowsiness Detection – Modern Applications and Methods**  
Fu, B., Boutros, F., Lin, C.-T., & Damer, N. (2024)  
[https://arxiv.org/abs/2408.12990](https://arxiv.org/abs/2408.12990)

**A Real-Time Embedded System for Driver Drowsiness Detection**  
Florez, R., et al. (2024)  
[https://www.mdpi.com/1424-8220/24/19/6261](https://www.mdpi.com/1424-8220/24/19/6261)

**Real-Time Tiredness Detection System Using Nvidia Jetson Nano and OpenCV**  
Florian, N., Popescu, D., & Hossu, A. (2024)  
[https://www.sciencedirect.com/science/article/pii/S1877050924018209](https://www.sciencedirect.com/science/article/pii/S1877050924018209)

---

## Acknowledgments

- **Dr. Tauseef Ur Rehman** - Project supervision and guidance
- **NVIDIA** - Jetson Nano Developer Kit and resources
- **OpenCV, TensorFlow, MediaPipe communities** - Tools and frameworks
- **NUST SEECS** - Research facilities and support

---

## License

MIT License

---

## Contact

**Mamona Sadaf**  
Email: msadaf.bee22seecs@seecs.edu.pk

**Repository**  
[https://github.com/Mamonasadaf/Driver-Fatigue-Detection](https://github.com/Mamonasadaf/Driver-Fatigue-Detection)

---

**This is an ongoing research project under active development**
