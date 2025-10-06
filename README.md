**Audio-Based Event Detection in Soccer (ELEC5305)**

Detect key soccer events (goal, foul, corner, etc.) from audio alone using deep learning on log-mel spectrograms. Compares a signal-based CNN/CRNN (RCNN) approach against an ASR+LLM baseline. 

**Why audio?**

Most systems lean on video and manual annotation, which is costly and slow. Audio carries rich cues—crowd surges, whistles, commentary excitement—and enables lighter, faster pipelines that can scale to long tournaments and real-time use.

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