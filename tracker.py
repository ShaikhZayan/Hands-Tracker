import cv2
import mediapipe as mp
import numpy as np
import math
import time

class HandTrackingModule:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.pTime = 0

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.hand_lm_style = self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2)  # Red landmarks
        self.hand_conn_style = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2)  # Green connections

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            if draw:
                for hand_landmarks in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS,
                                            self.hand_lm_style, self.hand_conn_style)
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if id in [4, 8]:  # Tips of the thumb and index finger
                            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)  # Slightly larger purple circle
                        else:
                            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)  # Smaller purple circle
            print("Hand detected")
        else:
            print("No hand detected")

        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return img

    def find_positions(self, img):
        landmarks_list = []
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                hand_landmarks = []
                for id, lm in enumerate(hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    hand_landmarks.append([id, cx, cy])
                landmarks_list.append(hand_landmarks)
        return landmarks_list

    def calculate_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y1 - y2)**2)

    def detect_click(self, landmarks, tip_id1, tip_id2, img):
        x1, y1 = landmarks[tip_id1][1], landmarks[tip_id1][2]
        x2, y2 = landmarks[tip_id2][1], landmarks[tip_id2][2]
        distance = self.calculate_distance(x1, y1, x2, y2)

        if distance < 30:  # Adjust the distance threshold for click detection here
            cv2.circle(img, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 15, (0, 255, 0), cv2.FILLED)  # Green circle if close enough
            cv2.putText(img, "Clicked", (10, img.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)  # Display "clicked"
            return True
        else:
            cv2.circle(img, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 10, (0, 0, 255), cv2.FILLED)  # Red circle if not close enough
            return False

def main():
    cap = cv2.VideoCapture(0)
    detector = HandTrackingModule()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        landmarks_list = detector.find_positions(img)

        for landmarks in landmarks_list:
            if 4 in [point[0] for point in landmarks] and 8 in [point[0] for point in landmarks]:
                x1, y1 = landmarks[4][1], landmarks[4][2]
                x2, y2 = landmarks[8][1], landmarks[8][2]
                distance = detector.calculate_distance(x1, y1, x2, y2)

                if distance < 100:  # Adjust the distance threshold for line drawing here
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red line for distance
                    detector.detect_click(landmarks, 4, 8, img)
                    cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)  # Larger purple circle for finger tips
                    cv2.circle(img, (x2, y2), 5, (255, 0, 255), cv2.FILLED)  # Larger purple circle for finger tips

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
