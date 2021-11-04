import random
import time

from cv2 import QRCodeDetector
from labels import labels
import gtts
from playsound import playsound
import audioread
import json
import numpy as np


class VoiceAssistant:
    current_letter = None
    finished_speaking_time = None
    has_suggested = False
    username = None
    c = 1.4

    def __init__(self):
        self.finished_speaking_time = time.time()

    def text_to_speech(self, text):
        tts = gtts.gTTS(text)
        tts.save('tmp/voice.mp3')
        playsound('tmp/voice.mp3', False)
        with audioread.audio_open('tmp/voice.mp3') as f:
            current_time = time.time()
            audio_duration = f.duration
            self.finished_speaking_time = current_time + audio_duration + 1

    def finished_speaking(self):
        return self.finished_speaking_time < time.time()

    def welcome(self, username):
        self.username = username
        self.text_to_speech('Welcome ' + username + '! I am your sign-language teacher. '
                            'I will tell you a letter and you can show me the corresponding sign. '
                            'Let\'s go!')

    def get_upc(self):
        upc_val = []
        avg_q = sum(Q) / len(Q)
        avg_q + c * np.sqrt(np.log(t/N))


    def suggest_letter(self):
        self.current_letter = random.choice(labels)
        self.text_to_speech(f'Do the sign-language sign for letter {self.current_letter.upper()}')
        self.has_suggested = True
        return self.current_letter


    def correct(self):
        self.text_to_speech('Correct!')
        self.has_suggested = False

    def incorrect(self):
        self.text_to_speech('Not quite there yet. Try again!')
