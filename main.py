import mediapipe as mp
import cv2
from tensorflow import keras
import numpy as np
from preprocessing import preprocess_keypoints
from voice_assistant import VoiceAssistant
from labels import labels

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

update_time = 10
save_time = 1000
prediction_freq = 2

assistant = VoiceAssistant()
model = keras.models.load_model('model')


def make_prediction_and_check_if_correct(hand_landmarks):
    preprocessed_input = preprocess_keypoints(hand_landmarks)
    prediction = model.predict(preprocessed_input)
    predicted_letter = labels[np.argmax(prediction)]
    print(f'Current letter: {assistant.current_letter}. Predicted letter: {predicted_letter}')
    return predicted_letter == assistant.current_letter


def run():
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

                if assistant.finished_speaking():
                    if assistant.has_suggested:
                        correct_sign = make_prediction_and_check_if_correct(hand_landmarks)
                        assistant.correct() if correct_sign else assistant.incorrect()
                    else:
                        assistant.suggest_letter()

            if key == ord(' '):
                break

            cv2.imshow('MediaPipe Hands', image)

    cap.release()


if __name__ == '__main__':
    run()
