import cv2
import mediapipe as mp
import numpy as np

class HandTrackingModule:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

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
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
            print("Hand detected")
        else:
            print("No hand detected")

        return img

    def find_positions(self, img, hand_num=0):
        landmarks_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_num]
            h, w, c = img.shape
            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks_list.append([id, cx, cy])
        return landmarks_list


def main():
    cap = cv2.VideoCapture(0)
    detector = HandTrackingModule()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        landmarks = detector.find_positions(img)
        if len(landmarks) != 0:
            print("Hand detected")
            for landmark in landmarks:
                print(f"Point {landmark[0]}: ({landmark[1]}, {landmark[2]})")
        else:
            print("No hand detected")

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
