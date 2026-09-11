

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from train import load_config, build_dataloader
from gnn_model import GraphEncoder
from bert_encoder import BERTEncoder
from fusion_model import GNNBERTFusionModel


@torch.no_grad()
def collect_z_and_valence(model, val_loader, valence_by_id: dict) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    zs, valences = [], []
    for batch in val_loader:
        out = model(batch["graph_batch"], batch["input_ids"], batch["attention_mask"])
        z_batch = out["z"].cpu().numpy()
        for z, track_id in zip(z_batch, batch["track_ids"]):
            if track_id in valence_by_id:
                zs.append(z)
                valences.append(valence_by_id[track_id])
    return np.array(zs), np.array(valences)


def generate_mood_tsne(cfg: dict, checkpoint_path: str, echonest_csv: str, out_path: str, seed: int = 42):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    genre_names = ckpt["genre_names"]

    graph_encoder = GraphEncoder(
        in_dim=ckpt["in_dim"], hidden_dim=cfg["model"]["gnn"]["hidden_dim"],
        num_layers=cfg["model"]["gnn"]["num_layers"], encoder_type=cfg["model"]["gnn"]["type"],
        dropout=cfg["model"]["gnn"]["dropout"],
    )
    text_encoder = BERTEncoder(model_name=cfg["model"]["bert"]["name"], freeze_base=cfg["model"]["bert"]["freeze_base"])
    model = GNNBERTFusionModel(
        graph_encoder=graph_encoder, text_encoder=text_encoder,
        graph_dim=graph_encoder.hidden_dim, text_dim=text_encoder.hidden_dim,
        num_tags=len(genre_names), predict_emotion=False,
    )
    model.load_state_dict(ckpt["model"])

    val_loader = build_dataloader(cfg, task=3, split="val")

    echonest = pd.read_csv(echonest_csv, index_col=0, header=[0, 1, 2])
    valence_col = [c for c in echonest.columns if c[-1] == "valence"][0]
    valence_by_id = echonest[valence_col].dropna().to_dict()
    print(f"Loaded {len(valence_by_id)} tracks with valence from {echonest_csv}")

    zs, valences = collect_z_and_valence(model, val_loader, valence_by_id)
    print(f"Matched {len(zs)} validation tracks with a valence score")

    if len(zs) < 5:
        print("Too few matched tracks for a meaningful t-SNE -- skipping. "
              "(echonest.csv only covers ~13k of FMA's 106k tracks, so overlap with "
              "your val split can be small, especially with --max-samples.)")
        return

    perplexity = min(30, max(2, len(zs) // 3))
    coords = TSNE(n_components=2, perplexity=perplexity, random_state=seed).fit_transform(zs)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=valences, cmap="coolwarm", s=25)
    plt.colorbar(scatter, label="Valence (mood: sad/negative \u2192 happy/positive)")
    plt.title("t-SNE of fused representation z (colored by mood/valence)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[mood t-sne] saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint_path = os.path.join(cfg["paths"]["checkpoints_dir"], "task3_cross_attention_model.pt")
    echonest_csv = os.path.join(os.path.dirname(cfg["dataset"]["fma"]["tracks_csv"]), "echonest.csv")
    out_path = os.path.join(cfg["paths"]["plots_dir"], "task3_tsne_mood.png")

    generate_mood_tsne(cfg, checkpoint_path, echonest_csv, out_path, seed=cfg["seed"])


if __name__ == "__main__":
    main()