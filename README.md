**Audio-Based Event Detection in Soccer (ELEC5305)**

Detect key soccer events (goal, foul, corner, etc.) from audio alone using deep learning on log-mel spectrograms. Compares a signal-based CNN/CRNN (RCNN) approach against an ASR+LLM baseline. 

**Important Points**
1. Motivation. Audio is cheap to process and carries strong cues (crowd surges, whistles, commentator excitement). It enables lighter, scalable detectors compared with video-heavy pipelines.

2. Scope. Work on SoccerNet-v2 matches (EPL subset), downloading only halves we actually need, extracting 16 kHz WAV, cutting event-centered clips (15 s/30 s), computing log-mel features, and training audio models. 
arXiv

3. Models. (1) RNN baseline over log-mels, and (2) pretrained wav2vec 2.0 backbone (frozen) with a light classification head. 
NeurIPS Proceedings

4. Status. Pipeline is complete end-to-end; initial baselines highlight heavy class imbalance. Next, we’ll fine-tune stronger AudioSet-pretrained backbones (PANNs / AST / HTS-AT) and rebalance data.

**Why audio?**

Most systems lean on video and manual annotation, which is costly and slow. Audio carries rich cues—crowd surges, whistles, commentary excitement—and enables lighter, faster pipelines that can scale to long tournaments and real-time use.

**Research Questions & Objectives**
Can audio-only models trained on broadcast soccer audio reach high macro-F1 on multi-class event detection, and what pretraining/backbone choices matter most under label imbalance?

Objectives.
- Build a space-efficient dataset pipeline (selective half downloads → WAV cache → clips → mel features/manifest).
- Establish mel-RNN and wav2vec 2.0-head baselines; validate feasibility and limits. 
NeurIPS Proceedings
- Improve accuracy with pretraining (AudioSet-based CNN/Transformers), data balancing, and augmentation (SpecAugment). 
- Report macro-F1, per-class PR, confusion matrices, and efficiency.
- Deliver a plug-and-play inference demo + model card for reuse.

**Prior Work**
- SoccerNet-v2 expands the original SoccerNet with ~300k manual annotations across 500 full matches and multiple tasks; it’s the de-facto benchmark suite for soccer understanding. 

- Audio pretraining:

  1. PANNs (AudioSet-pretrained CNNs; Wavegram-Logmel-CNN) transfer well across audio tasks. 
  2. AST (Audio Spectrogram Transformer) achieves SOTA on AudioSet/ESC-50 with transformer encoders on log-mels. 
  3. HTS-AT introduces hierarchical token-semantic transformers, strong for classification & localization. 
  4. wav2vec 2.0 learns powerful speech representations self-supervised; we repurpose as a frozen audio encoder for event cues. 

- Augmentation. SpecAugment time/frequency masking improves robustness for spectrogram models. 
- Commentary ASR resources. SoccerNet-Echoes provides Whisper-transcribed commentary JSONs, useful for ASR-assisted pipelines or weak supervision. 
- Early audio-sports: classic work showed crowd-response cues correlate with key events. 

Gap this repo targets: reproducible audio-only detectors tailored to SoccerNet, comparing classic mel-RNN baselines with strong pretrained encoders under label imbalance—with lightweight, selective data handling.

**Project goals**

1. Build an audio-only event detector on SoccerNet V2 broadcast audio. 
2. Train with 15s/30s event-centered windows and log-mel spectrograms. 
3. Compare spectrogram CNN/CRNN vs. ASR→LLM baseline.
4. Evaluate with Precision, Recall, F1 and confusion matrices; report efficiency.

**Repo Structure**
sn_audio_work/
  full_match_audio_wav/     # half-level WAV cache (from MKV) — H1/H2 at 16 kHz
  clips/
    win15s/                 # per-event clips (WAV)
    win30s/
  mel64/
    win15s/                 # per-event features (64xT .npy, log-mel z-scored)
    win30s/
  dataset_index.csv         # manifest used by the RCNN trainer
notebooks/
  00_download_labels.ipynb  # pull label JSONs only
  01_selective_audio.ipynb  # EPL-only: half-level MKV→WAV on demand
  02_make_features.ipynb    # cut clips, compute mel, write dataset_index.csv
  03_train_rcnn.ipynb       # RCNN (Conv+BiGRU) training/eval
README.md

**Data (SoccerNet V2)**

Get access to SoccerNet V2 and labels (small).

In 00_download_labels.ipynb download only label JSONs for all splits.

In 01_selective_audio.ipynb, we restrict to england_epl and download MKV only for halves you actually need, then convert to 16kHz WAV and optionally delete MKVs to save space.

**Results**
