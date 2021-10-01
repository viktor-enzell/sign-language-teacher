import cv2
import time
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

img_name = 'tmp/opencv_frame.png'
cam_width, cam_height = 640, 480
img_display_size = 360
img_size = 64

img_freq = 1
labels_short = ['a', 'b', 'c', 'd', 'e', '']
labels_long = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
          'w', 'x', 'y', 'z', '', ' ', 'del']

model = keras.models.load_model('models/CNN')


def make_prediction():
    img = preprocess_input(img_to_array(load_img(img_name, target_size=(img_size, img_size))))
    img = img.reshape(1, img_size, img_size, 3)

    prediction = model.predict(img)

    prediction_label = labels_short[np.argmax(prediction)]
    print(f'Predicted letter: {prediction_label}')

    return prediction_label


def run():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('Sign Language Teacher')

    cam.set(3, cam_width)
    cam.set(4, cam_height)

    latest_prediction = ''
    latest_img_time = time.time()
    while True:
        ret, frame = cam.read()
        if not ret:
            print('Failed to grab frame')
            break

        # Only capture a square area of size img_display_size
        frame = frame[int(cam_height / 2 - img_display_size / 2):int(cam_height / 2 + img_display_size / 2),
                int(cam_width / 2 - img_display_size / 2):int(cam_width / 2 + img_display_size / 2)]

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print('Escape hit, closing...')
            break
        if time.time() - latest_img_time > img_freq:
            # Take an image every 2 seconds
            latest_img_time = time.time()
            cv2.imwrite(img_name, frame)
            print('--------------\n'
                  'Image captured')

            latest_prediction = make_prediction()

        cv2.putText(
            frame, f'Letter: {latest_prediction}',
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 1
        )

        cv2.imshow('Sign Language Teacher', frame)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
