# AI-based Obstacle Detection and UAV Flight Control

This repository contains the source code developed for the Bachelor Thesis:

**AI-based technique for obstacle detection and UAV flight control**

The system implements a reactive obstacle avoidance prototype for a low-cost UAV using monocular RGB vision, YOLO object detection, remote processing and SDK/API-based actuation.

## Architecture

Perception → Decision → Actuation

## Main components

- UAV platform: DJI Tello / Tello-like UAV
- Camera: monocular RGB camera
- Processing: remote computer
- Communication: WiFi / UDP
- Programming language: Python
- Main libraries: OpenCV, djitellopy, Ultralytics YOLO
- Detection model: YOLO11n
- Decision logic: bounding box area, central safety region, detection persistence and obstacle-clear logic
- Actuation: conservative low-speed SDK/API commands

## Repository contents

- `tfg_v6_5.py`: final implementation used for the experimental prototype.
- `tfg_v6_5_line_by_line_guide.pdf`: line-by-line explanation of the code.
- `requirements.txt`: Python dependencies.

## Safety note

This project is an experimental reactive obstacle avoidance prototype. It is not a certified autonomous navigation system and must only be tested under controlled conditions with manual supervision.
