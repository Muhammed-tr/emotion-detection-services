Speech Emotion Recognition API
This project is a Flask-based web application for recognizing emotional states from audio files. It uses machine learning models powered by TensorFlow to analyze speech patterns and identify various emotions such as fear, anger, neutrality, happiness, sadness, and surprise.

Features
Upload audio files (supports WAV, MP3, OGG formats).
Process audio to extract MFCC features.
Predict emotional states using a pre-trained deep learning model.
Return emotion prediction results in JSON format.
Prerequisites
Before you can run this application, you'll need the following installed:

Python 3.8 or higher
Flask
Librosa
TensorFlow 2.x
Numpy

curl -X POST -F 'file=@path_to_your_audio_file.wav' http://127.0.0.1:5000/upload-audio
