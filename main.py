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

assistant = VoiceAssistant()
model = keras.models.load_model('model')

update_time = 10


def check_letter(hand_landmarks):
    # Predict the sign letter in the image
    preprocessed_input = preprocess_keypoints(hand_landmarks)
    prediction = model.predict([preprocessed_input])
    predicted_letter = labels[np.argmax(prediction)]

    # Compare the predicted letter with the suggested letter
    print(f'Current letter: {assistant.current_letter}. Predicted letter: {predicted_letter}')
    return predicted_letter == assistant.current_letter


def run():
    assistant.welcome()
    # For webcam input
    camera = cv2.VideoCapture(0)

    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while camera.isOpened():
            success, image = camera.read()
            if not success:
                print('Failed to read from camera. Exiting!')
                break
            # Flip the image horizontally for a later selfie-view display,
            # and convert the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable
            # to pass by reference.
            image.flags.writeable = False
            results = hands.process(image)
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            key = cv2.waitKey(update_time)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            # Wait with next action until assistant has finished speaking
            if assistant.finished_speaking():
                if assistant.has_suggested:
                    # Compare the suggested letter with the sign letter on the image
                    if results.multi_hand_landmarks:
                        correct_sign = check_letter(hand_landmarks)
                        assistant.correct() if correct_sign else assistant.incorrect()
                else:
                    assistant.suggest_letter()

            # Close window if space or escape key is pressed
            if key == ord(' ') or key == 27:
                print('Key pressed. Exiting')
                break

            cv2.imshow('Sign Language Teacher', image)

    camera.release()


if __name__ == '__main__':
    run()
