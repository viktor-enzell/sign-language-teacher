import cv2
import time
from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

img_name = 'img/opencv_frame.png'
img_width, img_height = 64, 64
img_freq = 1
labels_short = ['a', 'b', 'c', 'd', 'e', '']
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
          'w', 'x', 'y', 'z', '', ' ', 'del']

model = keras.models.load_model('models/CNN')

feature_extractor = applications.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(img_width, img_height, 3)
)


def make_prediction():
    img = preprocess_input(img_to_array(load_img(img_name, target_size=(img_width, img_height))))
    img = img.reshape(1, img_width, img_height, 3)

    # features = feature_extractor.predict(img)
    prediction = model.predict(img)

    prediction_label = labels_short[np.argmax(prediction)]
    print(f'Predicted letter: {prediction_label}')

    return prediction_label


def run():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('test')

    cam.set(3, 640)
    cam.set(4, 480)

    latest_prediction = ''
    latest_img_time = time.time()
    while True:
        ret, frame = cam.read()
        if not ret:
            print('failed to grab frame')
            break
        # TODO: look up if the frame can be cropped instead of stretched
        frame = cv2.resize(frame, (img_width * 6, img_height * 6), interpolation=cv2.INTER_AREA)

        cv2.putText(
            frame, f'Letter: {latest_prediction}',
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 1
        )

        cv2.imshow('Sign Language Teacher', frame)

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

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
