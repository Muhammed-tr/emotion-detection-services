from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename  # Import secure_filename
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import os
from datetime import datetime

app = Flask(__name__)

# Load the model
model = load_model("model3.h5")
CAT6 = ['fear', 'angry', 'neutral', 'happy', 'sad', 'surprise']

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  # Secure the filename
        filepath = os.path.join('audio', filename)
        file.save(filepath)
        return analyze_audio(filepath)
    return jsonify({"error": "Unsupported file type"}), 400

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['wav', 'mp3', 'ogg']

def analyze_audio(audio_path):
    try:
        # Processing audio file
        y, sr = librosa.load(audio_path, sr=None)  # Load with original sample rate
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        if mfccs.shape[1] > model.input_shape[-1]:
            mfccs = mfccs[:, :model.input_shape[-1]]
        elif mfccs.shape[1] < model.input_shape[-1]:
            pad_width = model.input_shape[-1] - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
        # Predict emotions
        mfccs = mfccs.reshape(1, *mfccs.shape)
        predictions = model.predict(mfccs)[0]
        emotions = {CAT6[i]: float(predictions[i]) for i in range(len(CAT6))}
        
        # Response with emotion scores
        return jsonify({"emotions": emotions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
