# 🧠 Human Posture Analysis using Deep Learning

This project implements a **Human Posture Recognition System** capable of detecting and classifying human postures such as **standing, sitting, walking, running, bending, lying, and stretching** using **deep learning models** (CNN + LSTM).

It supports **three types of input sources**:
- 🖼️ Image
- 🎥 Video
- 📷 Live Camera Stream

---

## 🚀 Features

- Real-time human posture detection
- Uses YOLOv8-Pose for keypoint extraction (CNN-based)
- LSTM / BiLSTM models for temporal pattern recognition
- Works on image, video, or webcam input
- Easily extendable with new posture classes or datasets

---

## 🧩 System Architecture

The project uses a **hybrid CNN–LSTM architecture**:

