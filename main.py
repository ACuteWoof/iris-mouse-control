import threading
import math

import cv2
import numpy as np
import pyautogui as pgui


def blur_filter(frame, d=5):
    kernel = np.ones((d, d), "float32") / (d ** 2)
    frame1 = cv2.GaussianBlur(frame, (d, d), 0)
    cv2.imshow("BLURRED IMAGE", frame1)
    return frame1


def erode_filter(frame, d=5):
    kernel = np.ones((d, d), np.uint8)
    frame1 = cv2.erode(frame, kernel, iterations=1)
    cv2.imshow("ERRODED IMAGE", frame1)
    return frame1


def get_pupils(main_frame, eye_frame, face_x, face_y, eye_x, eye_y, location_list):
    kernel = np.ones((3, 3), np.uint8)

    ret, binary = cv2.threshold(eye_frame, 60, 255, cv2.THRESH_BINARY_INV)
    width, height = binary.shape
    binary = binary[int(0.4 * height) : height, :]
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    dilate = cv2.morphologyEx(opening, cv2.MORPH_DILATE, kernel)
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        cnt = contours[0]
        m1 = cv2.moments(cnt)
        cx1 = int(m1["m10"] / m1["m00"])
        cy1 = int(m1["m01"] / m1["m00"])
        cropped_image_pixel_length = int(0.4 * height)
        center = (
            int(cx1 + face_x + eye_x),
            int(cy1 + face_y + eye_y + cropped_image_pixel_length),
        )
        cv2.circle(main_frame, center, 2, (0, 255, 0), 2)
        location_list.append(center)
        return center


def verify_movement(movement_logger_list, movement_info):
    if len(movement_logger_list) == 0:
        return True

    x_movement_rate = movement_logger_list[-1][0] - movement_info[0]
    y_movement_rate = movement_logger_list[-1][1] - movement_info[1]
    valid_movement_rate = 5
    if x_movement_rate >= valid_movement_rate or y_movement_rate >= valid_movement_rate:
        return True

    else:
        return False


def calculate_movement_length(location_list, movement_logger_list):
    try:
        x = location_list[-2][0] - location_list[-1][0]
        y = location_list[-2][1] - location_list[-1][1]
        x = (x//10)*10
        y = (y//10)*10
        min_movement_rate = 5
        x_piped = abs(x)
        y_piped = abs(y)
        if x_piped == x and y_piped == y:
            return False
        if x_piped > min_movement_rate or y_piped > min_movement_rate:
            is_movement_valid = verify_movement(movement_logger_list, (x*5, y*5))
            if is_movement_valid:
                print(f"{(x, y)} -> {(x*5, y*5)} |-> {(x_piped, y_piped)} -> {(x_piped*5, y_piped*5)}")
                movement_logger_list.append((x*5, y*5))
                return True
    except Exception as e:
        print(e)


def main():
    location_list = []
    movement_logger_list = []
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    while True:
        ret, frame = cap.read()

        if ret == False:
            break

        # Display blurred image
        blurred_frame = blur_filter(frame, d=9)

        # Make the image grayscale so it's convenient for the algorithms
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect all faces and loop through it, draw rectangles on the faces and detect eyes
        faces = face_cascade.detectMultiScale(gray, 1.5, 3, 30)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
            roi_face_gray = gray[y : y + w, x : x + w]
            roi_face_color = frame[y : y + h, x : x + w]
            eyes = eye_cascade.detectMultiScale(roi_face_gray, 1.3, 5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(
                    roi_face_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5
                )
                roi_eye_gray = roi_face_gray[ey : ey + ew, ex : ex + ew]
                roi_eye_color = roi_face_color[ey : ey + eh, ex : ex + ew]
                # apply blur
                blurred_eye = blur_filter(roi_eye_gray)
                # apply the erode filter
                eroded_eye = erode_filter(roi_eye_gray)
                # get cirlces
                pupils = get_pupils(frame, eroded_eye, x, y, ex, ey, location_list)
                should_i_move = calculate_movement_length(location_list, movement_logger_list)
                if should_i_move == True:
                    try:
                        pgui.move(-movement_logger_list[-1][0], movement_logger_list[-1][1], duration=0)
                    except pgui.FailSafeException:
                        continue
                else:
                    continue


        cv2.imshow("Woof's eye for mouse app", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
