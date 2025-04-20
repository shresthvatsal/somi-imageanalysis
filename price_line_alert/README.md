# Screen Capture and Analysis Project

## Overview
This project captures the screen at regular intervals, analyzes the images for intersections using OpenCV, and sends alerts via email.

## Directory Structure
- `images/`: Stores screenshots.
- `scripts/`: Contains Python code for screen capture, analysis, and alerting.
- `logs/`: Optional logs for debugging.
- `config/`: Store credentials if needed.

## Requirements
Install the following Python libraries:
- pyautogui
- cv2 (OpenCV)
- numpy
- smtplib

## How to Run
1. Modify the email configuration in `send_alert.py`.
2. Execute `main.py`.
