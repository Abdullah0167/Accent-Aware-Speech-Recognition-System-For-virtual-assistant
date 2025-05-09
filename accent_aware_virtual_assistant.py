"""
Accent-Aware Speech Recognition System
Using Deep Learning and Speaker Adaptation Techniques
for Virtual Assistant
"""

import torch
import torchaudio
import librosa
import os
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.pretrained import EncoderClassifier
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
print("[INFO] Loading models...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

speaker_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec"
)

# Audio preprocessing
def preprocess_audio(file_path, target_sr=16000):
    waveform, sr = librosa.load(file_path, sr=target_sr)
    return waveform

# Transcription using Wav2Vec2
def transcribe(audio_array):
    input_values = processor(audio_array, return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

# Extract speaker embedding
def extract_speaker_embedding(file_path):
    signal, fs = torchaudio.load(file_path)
    embeddings = speaker_encoder.encode_batch(signal)
    return embeddings.squeeze().detach().cpu().numpy()

# REST API Endpoint
@app.route("/transcribe", methods=["POST"])
def api_transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    file = request.files["audio"]
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    try:
        audio_array = preprocess_audio(filename)
        transcription = transcribe(audio_array)
        embedding = extract_speaker_embedding(filename)

        return jsonify({
            "transcription": transcription,
            "speaker_embedding": embedding.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the server
if __name__ == "__main__":
    print("[INFO] Starting Flask server on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)
