# ⚽ Audio-Based Event Detection in Soccer Matches (ELEC5305 Project)

This project explores how **deep learning models** can detect key soccer match events — like **goals, fouls, and corners** — using **audio signals alone**, without relying on video footage or manual annotation.

Built for the [ELEC5305](https://www.sydney.edu.au/units/ELEC5305) course, it implements and compares multiple modeling approaches (RNN, Wav2Vec2, AST) on the **SoccerNet V2** dataset to demonstrate the untapped potential of audio for real-time sports analytics.

---

## 🎯 Objectives

- Build a lightweight, audio-only pipeline for soccer event detection.
- Benchmark **signal-based deep learning models** vs **text-based ASR pipelines**.
- Demonstrate the viability of models like **Wav2Vec2** and **AST** (Audio Spectrogram Transformer) in sports analytics.
- Enable faster, scalable, and more accessible **automated soccer analytics** tools for broadcasters, coaches, and fans.

---

## 📦 Features

- 🔉 Extracts and segments **broadcast audio** from SoccerNet matches (EPL subset).
- 🧠 Implements multiple models:
  - RNN (baseline)
  - **Wav2Vec2** (pretrained on raw waveforms, frozen & fine-tuned variants)
  - **AST** (AudioSet pretrained transformer, fine-tuned)
- 🔬 Full support for log-mel spectrograms, data augmentation (noise, pitch/time shift)
- 📊 Evaluation via **Precision, Recall, F1**, confusion matrices, macro/micro scores
- 💾 Self-healing dataset builder with automatic clip generation and indexing
- ✅ Clean reproducible pipelines: training configs, seeds, logging, and checkpoints
- 🖥️ Inference notebook and command-line demo

---
📈 Results
| Model        | Accuracy  | Macro-F1 | Notes                       |
| ------------ | --------- | -------- | --------------------------- |
| RNN Baseline | 2.3%      | 0.5      | High class imbalance impact |
| Wav2Vec2     | 23%       | 0.25     | Pretrained, frozen layers   |
| AST          | 42%       | 0.45     | Pretrained, frozen layers   |
| AST (FT)     | **55%**   | **0.53** | AudioSet weights, tuned     |

🔬 Research Impact

This work demonstrates that audio-only pipelines can rival video-based systems for detecting critical sports events. It opens up new frontiers for low-latency, hardware-light, and scalable sports analytics using deep learning.

📚 References

SoccerNet V2 Dataset: https://silviogiancola.github.io/SoccerNetv2/
Wav2Vec2: https://arxiv.org/abs/2006.11477
AST: https://arxiv.org/abs/2104.01778
Whisper ASR: https://github.com/openai/whisper

👤 Author
Marcellus Ray Gunawan
Student ID: 520038655
