🛡️ Audio Fraud Sentinel
AI-Powered Real-Time Voice Integrity & Fraud Detection Framework
Audio Fraud Sentinel is a comprehensive defense system designed to detect AI-generated, cloned, replayed, or manipulated voices during live calls. By analyzing acoustic artifacts, speaker identity, and behavioral prosody in real-time, the system converts multiple signals into an actionable Impersonation Risk Score (0-100).
🚀 Overview
The system processes live audio in 3-5 second sliding windows, performing multi-layer analysis to ensure the voice on the other end is both human and who they claim to be.
Key Features
AI-Voice Detection: Identifies synthetic artifacts using Wav2Vec2 architecture.
Speaker Verification: Compares live voice embeddings against registered "Master" samples using ECAPA-TDNN.
Prosody Analysis: Detects anomalies in jitter, shimmer, and pitch variance common in synthetic speech.
Contextual Fraud Engine: Fuses ML scores with metadata (transaction value, caller ID) via an XGBoost model.
Explainable Alerts: Provides a detailed breakdown of why a call was flagged as high-risk.
🏗️ System Architecture
code
Mermaid
graph TD
    A[Live Audio Input] --> B[Streamlit Dashboard]
    B --> C[3-5s Audio Chunking]
    C --> D[Preprocessing: VAD/Noise Removal]
    D --> E{Multi-Layer Analysis}
    E --> F[AI-Voice Detection - Wav2Vec2]
    E --> G[Speaker Verification - ECAPA-TDNN]
    E --> H[Prosody Analysis - Jitter/Shimmer]
    F & G & H --> I[Contextual Fusion Engine]
    I --> J[XGBoost Risk Scorer]
    J --> K[Real-Time Dashboard Update]
    K --> L{Risk Level}
    L -->|0-30| M[LOW: Continue]
    L -->|30-80| N[HIGH: Warning]
    L -->|80-100| O[CRITICAL: Escalate]
📂 Project Structure
code
Text
/audio-fraud-sentinel
│
├── .streamlit/
│   └── config.toml          # Custom UI theme and font settings
│
├── core/                    # The ML "Brain"
│   ├── __init__.py
│   ├── preprocessor.py      # Audio normalization, VAD, and chunking logic
│   ├── detectors.py         # AI-voice detection & Speaker verification classes
│   └── prosody.py           # Signal processing for Jitter, Shimmer, and Pitch
│
├── models/                  # Pre-trained Model Weights
│   ├── synthetic_model.pth  # Wav2Vec2 weights for fake detection
│   ├── speaker_enc.onnx     # ECAPA-TDNN for identity verification
│   └── risk_engine.pkl      # Trained XGBoost model for risk fusion
│
├── data/                    # Storage
│   └── registry.json        # Database of authorized user voice embeddings
│
├── utils/                   # Logic Helpers
│   └── risk_calculator.py   # Aggregates ML scores into the final 0-100 score
│
├── app.py                   # Main Streamlit Entry Point (Frontend & Logic)
├── requirements.txt         # Project Dependencies
└── README.md                # Documentation
🛠️ Detailed Component Breakdown
1. ML Core (core/)
detectors.py: Implements the SyntheticDetector (returning "Fake Probability") and the SpeakerVerifier (performing cosine similarity against historical voice prints).
prosody.py: Uses Librosa to measure rhythmic and spectral characteristics. It flags voices that are "too perfect" or lack natural human micro-variations (Jitter/Shimmer).
2. Risk Engine (utils/risk_calculator.py)
This component implements the Fusion Model. It takes inputs from all detectors and applies a weighted XGBoost algorithm to generate a final risk percentage.
Input Vector: [AI_Probability, Speaker_Similarity, Prosody_Anomaly, Contextual_Weight]
Output: Final Risk Score (0-100).
3. Streamlit Dashboard (app.py)
The UI is divided into three functional zones:
Enrollment: Upload or record a baseline sample to register a "Trusted Speaker."
Live Monitoring: A real-time waveform and line chart showing the risk trend as the call progresses.
Explainability Panel: A detailed breakdown of detected reasons (e.g., "High probability of synthetic artifacts detected").
🚦 Getting Started
Prerequisites
Python 3.9+
FFmpeg (for audio processing)
Installation
Clone the repository:
code
Bash
git clone https://github.com/your-username/audio-fraud-sentinel.git
cd audio-fraud-sentinel
Install dependencies:
code
Bash
pip install -r requirements.txt
Run the application:
code
Bash
streamlit run app.py
📊 Risk Classification
Score	Level	Action Required
0–30	LOW	No action; normal verification.
30–60	MEDIUM	Monitor closely; additional caution.
60–80	HIGH	Secondary verification (MFA) recommended.
80–100	CRITICAL	Immediate Escalation; Block Transaction.
🛡️ Privacy & Ethics
This framework is designed with Privacy-by-Design principles. It prefers edge-based processing and only stores voice "embeddings" (mathematical representations) rather than raw audio files to ensure user data protection.
