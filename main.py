import cv2
import numpy as np

<<<<<<< HEAD
min_contour_width = 100
min_contour_height = 100
offset = 6
line_height = 351
matches = []
vehicles = 0
algo = cv2.createBackgroundSubtractorMOG2()


def get_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

=======
# Function to get centroid
def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
>>>>>>> 4af905f4e989e316660ed743739fecf3f01771da
    cx = x + x1
    cy = y + y1
    return cx, cy

<<<<<<< HEAD

cap = cv2.VideoCapture('23.mp4')

while True:
    ret, frame1 = cap.read()
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 10)
    apply_gray = algo.apply(blur)
    dilate = cv2.dilate(apply_gray, np.ones((1, 1)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    dilate2 = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilate3 = cv2.morphologyEx(dilate2, cv2.MORPH_CLOSE, kernel)
    counter, h = cv2.findContours(dilate3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, line_height), (1200, line_height), (255, 127, 0), 3)

    for (i, c) in enumerate(counter):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_contour_width) and (h >= min_contour_height)
        if not validate_counter:
            continue
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        circle = get_center(x, y, w, h)
        matches.append(circle)
        cv2.circle(frame1, circle, 4, (0, 0, 255), -1)
        for (x, y) in matches:
            if(line_height + offset) > y > (line_height - offset):
                vehicles += 1
                matches.remove((x, y))

            print("Number of Cars: "+str(vehicles))

    cv2.putText(frame1, "Number of cars: "+str(vehicles), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Fr", frame1)
    #cv2.imshow("dlt", dilate3)
    #cv2.imshow("gray", grey)
    #cv2.imshow("blru", blur)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
=======
# Video capture
cap = cv2.VideoCapture('xd.mp4')

# Load the car classifier
car_cascade = cv2.CascadeClassifier('cars.xml')

# Variables for vehicle counting
cars_detected = 0
line_height = 450  # Adjust the base line height as needed
offset = 10  # Adjust the offset as needed

# Calculate upper and lower bounds for counting
upper_bound = line_height - offset
lower_bound = line_height + offset

# Data structure to store the positions of detected cars
detected_car_positions = []

# Create a kernel for dilation
kernel = np.ones((5, 5), np.uint8)

# Loop to process each frame of the video
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (1, 1), 1)

    # Detect cars in the blurred frame
    cars = car_cascade.detectMultiScale(blurred, 1.1, 3)

    # Apply dilation to the binary mask of detected cars
    binary_mask = np.zeros_like(gray)
    for (x, y, w, h) in cars:
        binary_mask[y:y+h, x:x+w] = 255  # Create a binary mask for cars

    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    for (x, y, w, h) in cars:
        # Check if the car has been detected before based on its position
        already_detected = False
        cx, cy = get_centroid(x, y, w, h)

        for prev_cx, prev_cy in detected_car_positions:
            if abs(cx - prev_cx) < 10 and abs(cy - prev_cy) < 10:
                already_detected = True
                break

        if not already_detected:
            # Draw a rectangle around the detected car
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate and draw the centroid
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Check if the car crosses the line with offset
            if lower_bound > cy > upper_bound:
                cars_detected += 1

            # Store the position of the detected car
            detected_car_positions.append((cx, cy))

    # Display the total number of detected cars
    cv2.putText(frame, "Total Cars Detected: " + str(cars_detected), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2)

    # Draw the horizontal line with offset
    cv2.line(frame, (0, lower_bound), (frame.shape[1], lower_bound), (0, 0, 255), 2)

    # Show the frame with car detection
    cv2.imshow('Video', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
>>>>>>> 4af905f4e989e316660ed743739fecf3f01771da
