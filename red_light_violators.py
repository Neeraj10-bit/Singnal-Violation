from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 models
traffic_light_model = YOLO('yolov8n.pt')  # Pre-trained YOLOv8 model for traffic light detection
number_plate_model = YOLO('yolov8n.pt')  # Replace with a custom-trained YOLOv8 model for number plate detection

# Open the webcam (use 0 for the default webcam)
cap = cv2.VideoCapture(0)

# Define the stop line position (y-coordinate)
stop_line_y = 400  # Adjust this based on your webcam feed

# Define HSV color ranges for red, yellow, and green traffic lights
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
lower_green = np.array([40, 50, 50])
upper_green = np.array([90, 255, 255])

# Function to detect the color of the traffic light
def detect_traffic_light_color(frame, x1, y1, x2, y2):
    # Crop the traffic light region
    traffic_light_roi = frame[y1:y2, x1:x2]

    # Convert to HSV color space
    hsv = cv2.cvtColor(traffic_light_roi, cv2.COLOR_BGR2HSV)

    # Create masks for red, yellow, and green
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Count the number of non-zero pixels in each mask
    red_pixels = cv2.countNonZero(mask_red)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    green_pixels = cv2.countNonZero(mask_green)

    # Determine the dominant color
    if red_pixels > yellow_pixels and red_pixels > green_pixels:
        return "Red"
    elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
        return "Yellow"
    elif green_pixels > red_pixels and green_pixels > yellow_pixels:
        return "Green"
    else:
        return "Unknown"

# Main loop
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Perform traffic light detection
    traffic_light_results = traffic_light_model(frame)

    # Loop through detected traffic lights
    for result in traffic_light_results:
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id == 9:  # Class ID 9 for traffic lights in COCO dataset
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = detect_traffic_light_color(frame, x1, y1, x2, y2)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Traffic Light: {color}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Check for red light violation
                if color == "Red":
                    # Perform vehicle detection (assuming vehicles are class ID 2 in COCO dataset)
                    vehicle_results = traffic_light_model(frame)
                    for vehicle_result in vehicle_results:
                        for vehicle_box in vehicle_result.boxes:
                            vehicle_class_id = int(vehicle_box.cls)
                            if vehicle_class_id == 2:  # Class ID 2 for cars in COCO dataset
                                vx1, vy1, vx2, vy2 = map(int, vehicle_box.xyxy[0])

                                # Check if the vehicle crosses the stop line
                                if vy2 > stop_line_y:
                                    # Draw bounding box for violating vehicle
                                    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 0, 255), 2)
                                    cv2.putText(frame, "Red Light Violation", (vx1, vy1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                                    # Capture the number plate
                                    number_plate_results = number_plate_model(frame)
                                    for np_result in number_plate_results:
                                        for np_box in np_result.boxes:
                                            npx1, npy1, npx2, npy2 = map(int, np_box.xyxy[0])
                                            # Draw bounding box for number plate
                                            cv2.rectangle(frame, (npx1, npy1), (npx2, npy2), (255, 0, 0), 2)
                                            cv2.putText(frame, "Number Plate", (npx1, npy1 - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                                            # Save the number plate region
                                            number_plate_roi = frame[npy1:npy2, npx1:npx2]
                                            cv2.imwrite("number_plate.jpg", number_plate_roi)

    # Display the frame
    cv2.imshow('Traffic Light and Red Light Violation Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()