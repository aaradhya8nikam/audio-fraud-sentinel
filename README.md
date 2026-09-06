############################################
Detailed Project Workflow (Streamlit Logic)
############################################
Audio Capture: The Streamlit app uses a component (like streamlit-webrtc or a file uploader) to receive audio.
Chunking: The app splits the incoming stream into 3–5 second chunks (as per your PDF).
Preprocessing: Each chunk is converted to a Mel-Spectrogram or MFCC (using librosa).
Parallel ML Scoring:
Synthetic Check: Wav2Vec2 predicts if the audio is AI-generated.
Speaker Check: ECAPA-TDNN compares the chunk against a saved "Master" embedding of the real user.
Behavioral Check: Script calculates "Prosody Anomalies" (Jitter/Shimmer).
Risk Aggregation: An XGBoost model (loaded via pickle) takes these scores + manual metadata (e.g., Transaction Value) and calculates the Final Risk Score (0-100).
Stateful UI Update: Streamlit uses st.empty() or st.metric() to update the dashboard live as each chunk is processed, showing the risk trend.
##########################
2. Updated File Structure
##########################
code
Text
/voca-shield-app
│
├── .streamlit/
│   └── config.toml          # UI Theme (colors, font)
│
├── models/                  # Stored ML Weights
│   ├── synthetic_model.pth  # Wav2Vec2/WavLM weights
│   ├── speaker_enc.onnx     # ECAPA-TDNN model
│   └── risk_engine.pkl      # Trained XGBoost model
│
├── core/                    # The ML "Engine"
│   ├── __init__.py
│   ├── preprocessor.py      # Noise removal, VAD, chunking logic
│   ├── detectors.py         # AI-voice & Speaker verification logic
│   └── prosody.py           # Jitter, shimmer, and pitch analysis
│
├── data/                    # Local storage
│   └── registry.json        # Stores "Real User" voice embeddings
│
├── utils/                   # Helper scripts
│   └── risk_calculator.py   # Aggregates scores into 0-100
│
├── app.py                   # MAIN STREAMLIT ENTRY POINT
└── requirements.txt         # Dependencies (streamlit, torch, librosa, etc.)
##########################
3. Detailed File Contents
##########################
A. core/detectors.py (The Brain)
Content: Contains classes for the two main models.
Logic:
SyntheticDetector: Loads Wav2Vec2 and has a function predict(audio_chunk) that returns a "Fake Probability."
SpeakerVerifier: Uses speechbrain or ECAPA-TDNN to extract embeddings and perform a cosine_similarity check.
B. core/prosody.py (Behavioral Analysis)
Content: Pure signal processing using Librosa.
Logic: Measures Jitter (variation in pitch) and Shimmer (variation in amplitude). Synthetic voices often have unnaturally low jitter/shimmer.
C. utils/risk_calculator.py (The Engine)
Content: This implements Section 8 of your PDF.
Logic: It takes inputs: [AI_prob, Speaker_sim, Prosody_anomaly, Context_score] and runs them through the XGBoost model to return the final 0–100 score.
D. app.py (The Streamlit UI)
Top Sidebar: Upload a "Reference Voice" to register the user.
Center Panel:
st.audio_input: To record or stream live audio.
st.line_chart: To show the Risk Trend (e.g., how the risk went from 12% to 91% over 25 seconds).
st.status: To display the "Critical/High/Low" labels.
