import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import deque

def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLOv8 live traffic management")
    parser.add_argument(
        "--webcam-resolution",
        default=[1960, 1080],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")
    box_annotator = sv.BoxAnnotator(
        thickness=10,
        text_thickness=1,
        text_scale=1
    )

    green_count = 10
    yellow_count = 5
    stack = deque()  # Stack to track processed quadrants
    vehicle_counts = [0, 0, 0, 0]  # Vehicle counts for each quadrant
    yellow_phase = False
    active_quadrant = None  # Current quadrant being processed

    while True:
        ret, frame = cap.read()

        # Split the frame into quadrants
        height, width, _ = frame.shape
        half_height, half_width = height // 2, width // 2
        quadrants = [
            frame[:half_height, :half_width],
            frame[:half_height, half_width:],
            frame[half_height:, :half_width],
            frame[half_height:, half_width:]
        ]

        # Detect vehicles in each quadrant
        for i, quadrant in enumerate(quadrants):
            result = model(quadrant, agnostic_nms=True)[0]
            detections = sv.Detections.from_yolov8(result)
            detections = detections[(detections.class_id == 2) | (detections.class_id == 3)]

            labels = [
                f"Vehicle {j+1}: {model.model.names[class_id]} {confidence:0.2f}"
                for j, (bbox, confidence, class_id, _) in enumerate(detections)
            ]
            quadrant = box_annotator.annotate(
                scene=quadrant,
                detections=detections,
                labels=labels
            )

            vehicle_counts[i] = len(detections)

            if i == 0:
                frame[:half_height, :half_width] = quadrant
            elif i == 1:
                frame[:half_height, half_width:] = quadrant
            elif i == 2:
                frame[half_height:, :half_width] = quadrant
            elif i == 3:
                frame[half_height:, half_width:] = quadrant

        # Ensure that the active quadrant completes its green and yellow phases
        if active_quadrant is None or yellow_phase is False and green_count == 0:
            # If no quadrant is active, or the current quadrant has finished green phase, select a new quadrant
            if len(stack) < 4:  # Process quadrants until all are visited
                unvisited_quadrants = [i for i in range(4) if i not in stack]
                max_vehicle_count = max([vehicle_counts[i] for i in unvisited_quadrants])
                active_quadrant = next(
                    i for i in unvisited_quadrants if vehicle_counts[i] == max_vehicle_count
                )
                stack.append(active_quadrant)  # Push the quadrant to the stack
                green_count = 10
                yellow_phase = False

            else:
                stack.clear()  # Reset the stack for a new cycle

        # Manage traffic signal phases
        if yellow_phase:
            color = (0, 255, 255)  # Yellow light
            signal_text = f"YELLOW {yellow_count}"
            yellow_count -= 1
            if yellow_count == 0:
                yellow_phase = False
                active_quadrant = None  # Reset to allow new selection
        else:
            color = (0, 255, 0)  # Green light
            signal_text = f"GREEN {green_count}"
            green_count -= 1
            if green_count == 0:
                yellow_phase = True
                yellow_count = 5

        # Display signals for all quadrants
        for i in range(4):
            if i == active_quadrant:
                display_text = signal_text
                text_color = (0, 255, 0) if not yellow_phase else (0, 255, 255)
            else:
                display_text = "RED"
                text_color = (0, 0, 255)

            if i == 0:
                position = (10, half_height - 10)
                cv2.rectangle(frame, (0, 0), (half_width, half_height), (0, 255, 0), 2)
            elif i == 1:
                position = (half_width + 10, half_height - 10)
                cv2.rectangle(frame, (half_width, 0), (width, half_height), (0, 255, 0), 2)
            elif i == 2:
                position = (10, height - 10)
                cv2.rectangle(frame, (0, half_height), (half_width, height), (0, 255, 0), 2)
            elif i == 3:
                position = (half_width + 10, height - 10)
                cv2.rectangle(frame, (half_width, half_height), (width, height), (0, 255, 0), 2)

            cv2.putText(frame, display_text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        cv2.imshow("Traffic Management", frame)

        if cv2.waitKey(1000) == 27:  # Exit on 'ESC' key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
