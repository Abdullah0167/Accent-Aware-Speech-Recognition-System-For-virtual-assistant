# Accent-Aware Speech Recognition System for Virtual Assistant

This project implements a **real-time accent-aware speech recognition system** using deep learning and speaker adaptation techniques. It's designed to improve the performance of voice-based virtual assistants by adapting to different speaker accents and voices.

## Features

- Real-time speech-to-text transcription using Wav2Vec2
- Accent and speaker adaptation using x-vector embeddings from SpeechBrain
- RESTful API using Flask for easy integration
- Upload audio and get transcribed text and speaker embedding
- Supports `.wav` files sampled at 16kHz

---

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Torchaudio
- Librosa
- SpeechBrain
- Flask

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/accent-aware-virtual-assistant.git
cd accent-aware-virtual-assistant
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, manually install:
```bash
pip install torch torchaudio librosa transformers speechbrain flask
```

3. **Run the server:**

```bash
python app/accent_aware_virtual_assistant.py
```

---

## API Usage

### Endpoint:
`POST /transcribe`

### Payload:
Send a `.wav` file using form-data.

**Example using curl:**
```bash
curl -X POST -F "audio=@sample.wav" http://localhost:5000/transcribe
```

### Response:
```json
{
  "transcription": "Turn on the lights in the kitchen.",
  "speaker_embedding": [0.0412, -0.0133, ...]
}
```

---

## File Structure

```
accent-aware-virtual-assistant/
│
├── app/
│   └── accent_aware_virtual_assistant.py
│
├── uploads/
├── pretrained_models/
│   └── spkrec/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## License

This project is licensed under the MIT License.
