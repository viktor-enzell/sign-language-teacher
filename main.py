import mediapipe as mp
import cv2
from preprocessing import preprocess_keypoints
from voice_assistant import VoiceAssistant
import pickle
import json
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
assistant = VoiceAssistant()
model = pickle.load(open('models/random_forest.sav', 'rb'))
update_time = 10


def get_letter(hand_landmarks):
    # Predict the sign letter in the image
    preprocessed_input = preprocess_keypoints(hand_landmarks)
    return model.predict([preprocessed_input])[0]


def check_letter(hand_landmarks):
    # Compare the predicted letter with the suggested letter
    predicted_letter = get_letter(hand_landmarks)
    print(f'Current letter: {assistant.current_letter}. Predicted letter: {predicted_letter}')

    return predicted_letter == assistant.current_letter


# Checking if user is already registered 
def get_user_attempts(username):
    with open('data.txt', 'r+') as json_file:
        data = json.load(json_file)
        if username in data:
            return data[username]
        else:
            # Initalize the dictionary with all the possible labels
            user_attempts = {'total': 0, 'a': [], 'b': [],
                             'c': [], 'd': [], 'e': [], 'f': [], 'g': []}

            return user_attempts


def run():
    username = input("Enter username: ")
    print("your username is " + username)

    assistant.welcome(username)
    # For webcam input
    camera = cv2.VideoCapture(0)
    user_attempts = get_user_attempts(username)
    need_solution = False

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

                        if correct_sign:
                            assistant.correct()
                            user_attempts['total'] += 1
                            stop = time.time()

                            # Registering the time the user used on the letter
                            user_attempts[assistant.current_letter].append(
                                stop - start)
                            print(user_attempts[assistant.current_letter])
                        else:
                            assistant.incorrect()
                            if start > 30:
                                need_solution = True
                else:
                    assistant.suggest_letter(user_attempts)
                    start = time.time()
                    need_solution = False

            # Close window if space or escape key is pressed
            if key == ord(' ') or key == 27:
                print('Key pressed. Exiting')
                break

            # Adding which letter the user should show to the camera
            x, y, w, h = 40, 30, 480, 110
            alpha = 0.7
            overlay = image.copy()
            image = image.copy()
            cv2.rectangle(overlay, (x, x), (x + w, y + h + x), (0, 0, 0), -1)
            cv2.putText(overlay, (f'Show letter: {assistant.current_letter}'), (x + int(w / 10), y + int(h / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, (
                f'Letter you are currently signing: {get_letter(hand_landmarks)[0]}') if results.multi_hand_landmarks else "Show a sign",
                        (x + int(w / 10), y + int(h / 1.5) + x),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # Apply the overlay
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            # Add a little image of the sign language alphabet when the user has tried for more than 30 seconds
            if need_solution:
                wanted_image = assistant.current_letter
                img_alphabet = cv2.imread("img/" + wanted_image + ".png")

                x_offset = 40
                y_offset = 200
                x_end = x_offset + img_alphabet.shape[1]
                y_end = y_offset + img_alphabet.shape[0]
                image[y_offset:y_end, x_offset:x_end] = img_alphabet

            cv2.imshow('Sign Language Teacher', image)

    camera.release()

    with open('data.txt', 'r+') as json_file:
        data = json.load(json_file)
        data[username] = user_attempts
        json_file.seek(0)
        json.dump(data, json_file, indent=4)

if __name__ == '__main__':
    run()
