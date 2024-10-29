"""
@author: Ritalin
problem: Elle resim çizebilmek?
"""
import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

drawing_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
cap = cv2.VideoCapture(0)
last_x, last_y = None, None

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x = int(hand_landmarks.landmark[8].x * frame.shape[1])
                y = int(hand_landmarks.landmark[8].y * frame.shape[0])

                thumb_up = hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y
                index_finger_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
                middle_finger_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y

                if middle_finger_up:
                    drawing_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                    last_x, last_y = None, None
                elif index_finger_up:
                    if last_x is not None and last_y is not None:
                        cv2.line(drawing_canvas, (last_x, last_y), (x, y), (255, 255, 255), 3)
                    last_x, last_y = x, y
                else:
                    last_x, last_y = None, None

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Parmak Takibi ile paint', frame)
        cv2.imshow('Paint', drawing_canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()