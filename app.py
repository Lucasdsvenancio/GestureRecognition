import csv
import cv2 as cv
import copy
import mediapipe as mp
import numpy as np
import itertools

from model import KeyPointClassifier

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def logging_csv(index, mode, landmark_list):
    if mode == 0:
        return
    elif mode == 1:
        if index >= 0:
            csv_path = 'model/keypoint_classifier/keypoint.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([index, *landmark_list])
    return

def select_mode(key, mode):
    number = -1
    if key == 49: # Changed modes to numbers and letters to themselves
        mode = 0
    if key == 50:
        mode = 1
    if key == 51: # To train default hand
        number = 0
    if 97 <= key <= 122:
        number = key - 96

    return number, mode

def draw_debug_info(image, mode, index):
    if mode != 0:
        cv.putText(image, 'Program mode: Logging Key Points', (10, 90),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            cv.LINE_AA)
        if 0 <= index <= 26:
            cv.putText(image, f'Num: {str(index)}', (10, 110),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv.LINE_AA)
    
    return image

def draw_hand_info(image, brect, handedness, hand_sign_text):
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = f'{info_text}: {hand_sign_text}'
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (114,109,85), 1, cv.LINE_AA)

    return image

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.85,
        min_tracking_confidence=0.6
    )

    keypoint_classifier = KeyPointClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    
    mode = 0

    while cap.isOpened():
        key = cv.waitKey(10)
        if key == 27:
            break

        index, mode = select_mode(key, mode)

        success, image = cap.read()
        if not success:
            continue
    
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                brect = calc_bounding_rect(debug_image, hand_landmarks)

                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                logging_csv(index, mode, pre_processed_landmark_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                mp_drawing.draw_landmarks(
                    debug_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(110,22,10), thickness=2, circle_radius=7),
                    mp_drawing.DrawingSpec(color=(50,110,121), thickness=2, circle_radius=3))
                
                debug_image = draw_hand_info(debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id])

        debug_image = draw_debug_info(debug_image, mode, index)

        cv.imshow('Alphabet Recognition w/ Mediapipe', debug_image)

    cap.release()
    cv.destroyAllWindows()