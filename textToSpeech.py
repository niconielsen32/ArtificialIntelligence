from gtts import gTTS
from playsound import playsound

audio = 'speech.mp3'
language = 'en'
speech = gTTS(text="Hey Guys! Thanks for watching", lang=language)

speech.save(audio)
playsound("C:/Users/nhoei/MachineLearningProject/speech.mp3")
