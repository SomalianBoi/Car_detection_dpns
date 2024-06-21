import cv2
import numpy as np

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

    cx = x + x1
    cy = y + y1
    return cx, cy


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

    #cv2.line(frame1, (25, line_height), (1200, line_height), (255, 127, 0), 3)

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

    #cv2.putText(frame1, "Number of cars: "+str(vehicles), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Fr", frame1)
    #cv2.imshow("dlt", dilate3)
    #cv2.imshow("gray", grey)
    #cv2.imshow("blru", blur)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
