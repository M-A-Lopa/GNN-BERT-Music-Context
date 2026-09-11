# gnn-bert-music-context

Hybrid **BERT + Graph Neural Network** system for music context understanding —
multi-label tagging, emotion regression, and cross-modal (audio-structure ↔ text)
retrieval, per the course assignment spec.

## Task roadmap

| Task | Model | Goal |
|------|-------|------|
| 1 (Easy) | BERT tag classifier | Multi-label tags from text (tags/captions/lyrics) |
| 2 (Medium) | GraphSAGE/GAT on segment/chord graphs | Genre/tag prediction from audio structure only |
| 3 (Hard) | GNN–BERT cross-attention fusion | Multi-label tags + optional valence/arousal |
| 4 (Advanced) | Contrastive dual-encoder | Audio-graph ↔ caption retrieval (MusicCaps) |

## Status

Task 4 is not done for now. 

## Setup

```bash
pip install -r requirements.txt
```

Edit `config.yaml` to point `data.raw_dir` at wherever you download the raw
datasets, then run preprocessing (once `src/audio_features.py` and
`src/graph_builder.py` are filled in) followed by `src/train.py --task 1`.

## Project structure

```
gnn-bert-music-context/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/            # FMA, MagnaTagATune, MusicCaps downloads
│   ├── processed/       # graphs, mel-spec, BERT caches
│   └── splits/          # train/val/test JSON
├── notebooks/
│   ├── eda.ipynb
│   └── demo_context.ipynb
├── src/
│   ├── audio_features.py   # mel, chroma, segmentation
│   ├── graph_builder.py    # chord + segment graphs
│   ├── bert_encoder.py
│   ├── gnn_model.py         # GraphSAGE / GAT
│   ├── fusion_model.py      # cross-attention GNN-BERT
│   ├── train.py
│   └── evaluate.py
├── results/
│   ├── metrics.json
│   ├── plots/
│   └── retrieval_examples/
└── report/
    └── final_report.pdf
```
