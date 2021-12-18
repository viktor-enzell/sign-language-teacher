import time

import gtts
from playsound import playsound
import audioread
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
                                                    'I will mention a letter and you show me the sign. '
                                                    'Let\'s go!')

    # Takes dictionary of attempts and returs the letter with the highest UCB
    def get_ucb(self, user):
        ucb_list = [0]
        key_list = list(user)

        for letter in user:
            if letter == 'total':
                t = user['total']
            else:
                times = user[letter]
                N = len(times)
                if N:
                    Q = sum(times) / (30 * N)
                    ucb = Q + self.c * np.sqrt(np.log(t) / N)
                    ucb_list.append(ucb)
                else:
                    ucb_list.append(30)
        return key_list[np.argmax(ucb_list)]

    # Based on dictionary of attempted times, reads and suggests a letter
    def suggest_letter(self, attempts):
        self.current_letter = self.get_ucb(attempts)
        self.text_to_speech(f'Do the sign for {self.current_letter.upper()}')
        self.has_suggested = True
        return self.current_letter

    # Reads correct
    def correct(self):
        self.text_to_speech('Correct!')
        self.has_suggested = False

    # Reads incorrect
    def incorrect(self):
        self.text_to_speech('Not quite there yet. Try again!')
