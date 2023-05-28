import cv2
import numpy as np
# import matplotlib.pyplot as plt


def make_coordinates(image, line_parameters):
    if line_parameters is None:
        return None

    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intersect(image, lines_for_calc):
    if lines is None or len(lines) == 0:
        return None

    left_fit = []
    right_fit = []
    for line in lines_for_calc:
        x1, y1, x2, y2 = line.reshape(4)
        parameter = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameter[0]
        intercept = parameter[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    if len(left_fit) > 0 and len(right_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)

        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])
    else:
        return None


def canny(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_canny = cv2.Canny(img_blur, 50, 150)
    return img_canny


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    # zeros array with the same amount of rows as cols
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines_for_action):
    line_on_image = np.zeros_like(image)
    if lines_for_action is not None:
        for line in lines_for_action:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_on_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_on_image


cap = cv2.VideoCapture("Resources/test2.mp4")
while cap.isOpened():
    success, frame = cap.read()
    if success:
        edges = canny(frame)
        cropped_image = region_of_interest(edges)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        if lines is not None and len(lines) > 0:
            averaged_lines = average_slope_intersect(frame, lines)
            if averaged_lines is not None:
                line_image = display_lines(frame, averaged_lines)
                combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
                cv2.imshow("result", combo_image)
            else:
                print("No valid lines found.")
        else:
            print("No averaged lines detected")
    else:
        print("End of video")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
