# YOLOv8 Live Traffic Management System

This repository contains a Python script that utilizes the YOLOv8 object detection model to create a live traffic management system. The system divides a live webcam feed into four quadrants, detects vehicles in each quadrant, and simulates traffic light control based on the vehicle density in each section.

---

## Features
- **Live Vehicle Detection:** Leverages the YOLOv8 object detection model to identify vehicles in real-time.
- **Traffic Signal Management:** Implements a round-robin traffic light algorithm based on vehicle density.
- **Dynamic Quadrant Analysis:** Splits the video feed into four quadrants for localized traffic monitoring.
- **Phased Signal Control:** Simulates green and yellow light phases for the active quadrant, while others remain on red.
- **Adaptive Cycle Reset:** Automatically resets the cycle once all quadrants are processed.

---

## Requirements

### Hardware Requirements
- Webcam (for live video feed).
- System with GPU support (recommended for optimal performance).

### Software Requirements
- Python 3.8+
- Required libraries:
  - `cv2` (OpenCV)
  - `argparse`
  - `numpy`
  - `ultralytics` (for YOLOv8)
  - `supervision`
  - `collections` (deque for managing active quadrants)

---

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/yolov8-traffic-management.git
   cd yolov8-traffic-management
2. Install required Python libraries:
   ```bash
   pip install -r requirements.txt
3. Download the YOLOv8 model weights and place them in the working directory:
   ```bash
   ultralytics.models.yolov8l.pt

## Usage

### Command Line Arguments
- This script supports the following optional arguments:
    `--webcam-resolution`: Set the resolution of the webcam feed. Default is `[1960, 1080]`.

### Run the Script
- Execute the script using the following command:
   ```bash
  python traffic_management.py --webcam-resolution 1280 720

## Code Breakdown
1. Argument Parsing
- Parses the webcam resolution using the argparse library.
   ```bash
  def parse_arguments():
      parser = argparse.ArgumentParser(description="YOLOv8 live traffic management")
      parser.add_argument("--webcam-resolution", default=[1960, 1080], nargs=2, type=int)
      return parser.parse_args()


2. YOLOv8 Initialization
- Loads the YOLOv8 model for vehicle detection.
- Configures the bounding box annotator for visualizing detections.
   ```bash
   model = YOLO("yolov8l.pt")
  box_annotator = sv.BoxAnnotator(thickness=10, text_thickness=1, text_scale=1)


3. Frame Splitting
- Splits the live webcam feed into four quadrants for localized analysis.
   ```bash
  quadrants = [
      frame[:half_height, :half_width],
      frame[:half_height, half_width:],
      frame[half_height:, :half_width],
      frame[half_height:, half_width:]
  ]

4. Vehicle Detection
- Uses YOLOv8 to detect vehicles (class IDs 2 and 3) in each quadrant.
- Annotates detected vehicles with labels.
   ```bash
  detections = sv.Detections.from_yolov8(result)
  detections = detections[(detections.class_id == 2) | (detections.class_id == 3)]

5. Traffic Signal Management
- Implements a prioritized round-robin system to activate the quadrant with the highest vehicle count.
- Manages green and yellow light phases for each quadrant.
   ```bash
  if yellow_phase:
      signal_text = f"YELLOW {yellow_count}"
      yellow_count -= 1
  else:
      signal_text = f"GREEN {green_count}"
      green_count -= 1

6. Display Signals
-Overlays the signal status (RED, GREEN, YELLOW) on each quadrant.
   ```bash
   cv2.putText(frame, display_text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

## Output
- The system displays a live feed with annotated quadrants showing vehicle counts.
- Traffic signal status is dynamically updated for each quadrant.

## Future Improvements
- Integrate traffic signal hardware for real-world deployment.
- Extend vehicle detection to include other object types (e.g., pedestrians, bicycles).
- Optimize YOLOv8 inference for improved performance on low-end systems.
- Add historical data logging for traffic analysis.

## Acknowledgements
- Ultralytics for the YOLOv8 model.
- Supervision library for enhancing visualizations.
