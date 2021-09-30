import random
import gtts
from playsound import playsound


class VoiceAssistant:
    labels_short = ['a', 'b', 'c', 'd', 'e']
    current_letter = None

    def text_to_speech(self, text):
        tts = gtts.gTTS(text)
        tts.save('voice.mp3')
        playsound('voice.mp3', False)

    def welcome(self):
        self.text_to_speech('Welcome! I am your sign-language teacher. '
                            'I will tell you a letter and you can show me the corresponding sign. '
                            'Let\'s go!')

    def suggest_letter(self):
        self.current_letter = random.choice(self.labels_short)
        self.text_to_speech(f'Do the sign-language sign for letter {self.current_letter}')
        return self.current_letter

    def check_letter(self, user_letter):
        if self.current_letter == user_letter:
            self.text_to_speech('Correct!')
            return True
        else:
            self.text_to_speech('Not quite there yet. Try again!')
            return False
