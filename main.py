import mediapipe as mp
import cv2
import time

from voice_assistant import VoiceAssistant

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

update_time = 10
save_time = 1000
prediction_freq = 2

labels_short = ['a', 'b', 'c', 'd', 'e']


def make_prediction_and_check_if_correct():
    return False


def run():
    assistant = VoiceAssistant()
    assistant.welcome()

    # For webcam input:
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        i = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print('Ignoring empty camera frame.')
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            key = cv2.waitKey(update_time)
            i = i + update_time
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            if key == ord(' '):
                break

            if assistant.finished_speaking():
                if assistant.has_suggested:
                    correct_sign = make_prediction_and_check_if_correct()
                    if correct_sign:
                        assistant.correct()
                    else:
                        assistant.incorrect()
                else:
                    assistant.suggest_letter()

            cv2.imshow('MediaPipe Hands', image)

    cap.release()


if __name__ == '__main__':
    run()
