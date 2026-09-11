from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import yaml
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from transformers import AutoTokenizer

import evaluate as ev
from bert_encoder import BERTEncoder, TagClassifierHead
from gnn_model import GraphEncoder, GraphTagHead
from fusion_model import GNNBERTFusionModel, EarlyConcatFusionModel
from datasets import (
    MagnaTagATuneDataset, collate_fn as mtt_collate_fn,
    FMAGraphDataset, FMAGraphTextDataset, fma_task3_collate_fn,
    FMAMelDataset,
)
from cnn_baseline import SimpleCNN


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_dataset_name(cfg: dict, task: int) -> str:
    overrides = cfg["dataset"].get("task_overrides", {})
    return overrides.get(task, cfg["dataset"]["default"])


def build_dataloader(cfg: dict, task: int, split: str = "train") -> DataLoader:
    dataset_name = resolve_dataset_name(cfg, task)
    fma_split_names = {"train": "training", "val": "validation", "test": "test"}  
    max_samples = cfg["dataset"].get("max_samples")  

    if dataset_name == "magnatagatune":
        mtt_cfg = cfg["dataset"]["magnatagatune"]
        split_file = os.path.join(cfg["data"]["splits_dir"], f"mtt_{split}_ids.tsv")

        dataset = MagnaTagATuneDataset(
            annotations_csv=mtt_cfg["annotations_csv"],
            split_ids_tsv=split_file,
            top_k_tags=cfg["dataset"]["top_k_tags"],
            mask_ratio=mtt_cfg["mask_ratio"],
            seed=cfg["seed"],
            max_samples=max_samples,
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["bert"]["name"])

        return DataLoader(
            dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=(split == "train"),
            collate_fn=lambda batch: mtt_collate_fn(
                batch, tokenizer, max_length=cfg["text"]["max_length"]
            ),
        )

    if dataset_name == "fma_medium":
        fma_cfg = cfg["dataset"]["fma"]
        fma_split = fma_split_names[split]
        graph_cache_dir = os.path.join(cfg["data"]["processed_dir"], "fma_graphs")

        if task == 2:
            dataset = FMAGraphDataset(
                tracks_csv=fma_cfg["tracks_csv"],
                audio_dir=fma_cfg["audio_dir"],
                subset=fma_cfg["subset"],
                split=fma_split,
                num_genre_classes=cfg["dataset"]["num_genre_classes"],
                sample_rate=cfg["dataset"]["sample_rate"],
                hop_length=cfg["audio_features"]["hop_length"],
                segment_seconds=cfg["dataset"]["segment_seconds"],
                similarity_threshold=cfg["graph"]["segment_similarity_threshold"],
                n_mels=cfg["audio_features"]["n_mels"],
                n_chroma=cfg["audio_features"]["n_chroma"],
                max_samples=max_samples,
                cache_dir=graph_cache_dir,
            )
            return PyGDataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=(split == "train"))

        if task == 3:
            dataset = FMAGraphTextDataset(
                tracks_csv=fma_cfg["tracks_csv"],
                audio_dir=fma_cfg["audio_dir"],
                raw_tracks_csv=fma_cfg["raw_tracks_csv"],
                subset=fma_cfg["subset"],
                split=fma_split,
                num_genre_classes=cfg["dataset"]["num_genre_classes"],
                sample_rate=cfg["dataset"]["sample_rate"],
                hop_length=cfg["audio_features"]["hop_length"],
                segment_seconds=cfg["dataset"]["segment_seconds"],
                similarity_threshold=cfg["graph"]["segment_similarity_threshold"],
                n_mels=cfg["audio_features"]["n_mels"],
                n_chroma=cfg["audio_features"]["n_chroma"],
                max_samples=max_samples,
                cache_dir=graph_cache_dir,
            )
            tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["bert"]["name"])
            return DataLoader(
                dataset,
                batch_size=cfg["training"]["batch_size"],
                shuffle=(split == "train"),
                collate_fn=lambda batch: fma_task3_collate_fn(
                    batch, tokenizer, max_length=cfg["text"]["max_length"]
                ),
            )

    raise NotImplementedError(
        f"No dataset loader implemented yet for task {task} (dataset='{dataset_name}', "
        f"split='{split}'). Download/preprocess the data first, then add a Dataset class "
        "and build it here."
    )


def make_optimizer(cfg: dict, scratch_modules=(), pretrained_modules=()) -> torch.optim.Optimizer:
    param_groups = []
    for m in scratch_modules:
        param_groups.append({"params": m.parameters(), "lr": cfg["training"]["lr_scratch"]})
    for m in pretrained_modules:
        param_groups.append({"params": m.parameters(), "lr": cfg["training"]["lr"]})
    return torch.optim.AdamW(param_groups, weight_decay=cfg["training"]["weight_decay"])


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"[checkpoint] saved to {path}")


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def single_label_predictions_to_onehot(y_probs: np.ndarray) -> np.ndarray:
    pred_idx = y_probs.argmax(axis=1)
    y_pred = np.zeros_like(y_probs)
    y_pred[np.arange(len(y_probs)), pred_idx] = 1.0
    return y_pred


def plot_f1_curves(history: list[dict], out_path: str, title: str):
    epochs = [h["epoch"] for h in history]
    plt.figure()
    plt.plot(epochs, [h["val_macro_f1"] for h in history], label="Macro-F1", marker="o")
    plt.plot(epochs, [h["val_micro_f1"] for h in history], label="Micro-F1", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved to {out_path}")


@torch.no_grad()
def collect_multilabel_predictions(forward_fn, dataloader, device):
    all_probs, all_labels = [], []
    for batch in dataloader:
        logits, labels = forward_fn(batch, device)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


# Task 1: BERT multi-label tag classifier 

def build_model_task1(cfg: dict, num_tags: int, device: torch.device):
    encoder = BERTEncoder(
        model_name=cfg["model"]["bert"]["name"],
        freeze_base=cfg["model"]["bert"]["freeze_base"],
    ).to(device)
    head = TagClassifierHead(hidden_dim=encoder.hidden_dim, num_tags=num_tags).to(device)
    return encoder, head


def run_step_task1(encoder, head, batch, loss_fn, device: torch.device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    cls_embedding, _ = encoder(input_ids, attention_mask)
    logits = head(cls_embedding)
    loss = loss_fn(logits, labels.float())
    return loss, logits


def train_task1(cfg: dict):
    device = get_device()
    print(f"[task1] Running on device: {device}")
    num_tags = cfg["dataset"]["top_k_tags"]
    encoder, head = build_model_task1(cfg, num_tags, device)
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = build_dataloader(cfg, task=1, split="train")
    val_loader = build_dataloader(cfg, task=1, split="val")
    tag_names = train_loader.dataset.tag_names

    def forward_eval(batch, dev):
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch["attention_mask"].to(dev)
        cls_embedding, _ = encoder(input_ids, attention_mask)
        return head(cls_embedding), batch["labels"]

    history = []
    for epoch in range(cfg["training"]["epochs"]):
        encoder.train()
        head.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, _ = run_step_task1(encoder, head, batch, loss_fn, device)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)

        encoder.eval()
        head.eval()
        y_true, y_probs = collect_multilabel_predictions(forward_eval, val_loader, device=device)
        y_pred = (y_probs > 0.5).astype("float32")
        f1_metrics = ev.precision_recall_f1_per_tag(y_true, y_pred)

        print(
            f"[task1] epoch {epoch + 1}/{cfg['training']['epochs']} "
            f"train_loss={train_loss:.4f} val_macro_f1={f1_metrics['macro_f1']:.4f} "
            f"val_micro_f1={f1_metrics['micro_f1']:.4f}"
        )
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_macro_f1": f1_metrics["macro_f1"],
            "val_micro_f1": f1_metrics["micro_f1"],
        })

    results_dir = cfg["paths"]["results_dir"]
    plot_f1_curves(
        history, os.path.join(cfg["paths"]["plots_dir"], "task1_f1_curves.png"),
        title="Task 1: MagnaTagATune tag classification",
    )
    save_json(history, os.path.join(results_dir, "task1_metrics.json"))
    save_checkpoint(
        {"encoder": encoder.state_dict(), "head": head.state_dict(), "tag_names": tag_names},
        os.path.join(cfg["paths"]["checkpoints_dir"], "task1_model.pt"),
    )

    encoder.eval()
    head.eval()
    examples = []
    with torch.no_grad():
        for i in range(min(5, len(val_loader.dataset))):
            item = val_loader.dataset[i]
            encoded = val_loader.collate_fn([item])
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            cls_embedding, _ = encoder(input_ids, attention_mask)
            probs = torch.sigmoid(head(cls_embedding))[0]
            predicted_tags = [tag_names[idx] for idx, p in enumerate(probs) if p > 0.5]
            true_tags = [tag_names[idx] for idx, v in enumerate(item["labels"]) if v > 0]
            examples.append({
                "input_text": item["text"],
                "true_tags": true_tags,
                "predicted_tags": predicted_tags,
            })
    save_json(examples, os.path.join(results_dir, "task1_example_predictions.json"))

    return encoder, head


# Task 2: GNN on segment/chord graphs

def build_model_task2(cfg: dict, in_dim: int, num_classes: int, device: torch.device):
    encoder = GraphEncoder(
        in_dim=in_dim,
        hidden_dim=cfg["model"]["gnn"]["hidden_dim"],
        num_layers=cfg["model"]["gnn"]["num_layers"],
        encoder_type=cfg["model"]["gnn"]["type"],
        dropout=cfg["model"]["gnn"]["dropout"],
    ).to(device)
    head = GraphTagHead(hidden_dim=encoder.hidden_dim, num_classes=num_classes).to(device)
    return encoder, head


def run_step_task2(encoder, head, batch, loss_fn, device: torch.device):
    batch = batch.to(device)
    _, graph_embedding = encoder(batch.x, batch.edge_index, batch.batch)
    logits = head(graph_embedding)
    loss = loss_fn(logits, batch.y.float())
    return loss, logits


def train_task2(cfg: dict):
    device = get_device()
    print(f"[task2] Running on device: {device}")
    train_loader = build_dataloader(cfg, task=2, split="train")
    val_loader = build_dataloader(cfg, task=2, split="val")

    first_batch = next(iter(train_loader))
    in_dim = first_batch.x.shape[1]
    num_classes = cfg["dataset"]["num_genre_classes"]

    encoder, head = build_model_task2(cfg, in_dim, num_classes, device)
    optimizer = make_optimizer(cfg, scratch_modules=[encoder, head])
    loss_fn = nn.BCEWithLogitsLoss()

    def forward_eval(batch, dev):
        batch = batch.to(dev)
        _, graph_embedding = encoder(batch.x, batch.edge_index, batch.batch)
        return head(graph_embedding), batch.y

    history = []
    for epoch in range(cfg["training"]["epochs"]):
        encoder.train()
        head.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, _ = run_step_task2(encoder, head, batch, loss_fn, device)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)

        encoder.eval()
        head.eval()
        y_true, y_probs = collect_multilabel_predictions(forward_eval, val_loader, device=device)
        y_pred = single_label_predictions_to_onehot(y_probs)  # genre is single-label, not independently thresholded
        f1_metrics = ev.precision_recall_f1_per_tag(y_true, y_pred)

        print(
            f"[task2] epoch {epoch + 1}/{cfg['training']['epochs']} train_loss={train_loss:.4f} "
            f"val_macro_f1={f1_metrics['macro_f1']:.4f} val_micro_f1={f1_metrics['micro_f1']:.4f}"
        )
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_macro_f1": f1_metrics["macro_f1"], "val_micro_f1": f1_metrics["micro_f1"],
        })

    plot_f1_curves(
        history, os.path.join(cfg["paths"]["plots_dir"], "task2_gnn_f1_curves.png"),
        title="Task 2: GNN genre classification",
    )
    save_json(history, os.path.join(cfg["paths"]["results_dir"], "task2_gnn_metrics.json"))
    save_checkpoint(
        {"encoder": encoder.state_dict(), "head": head.state_dict(),
         "genre_names": train_loader.dataset.genre_names, "in_dim": in_dim},
        os.path.join(cfg["paths"]["checkpoints_dir"], "task2_gnn_model.pt"),
    )

    cnn_model, cnn_history = train_task2_cnn_baseline(cfg, device)
    comparison = {
        "gnn": {"macro_f1": history[-1]["val_macro_f1"], "micro_f1": history[-1]["val_micro_f1"]},
        "cnn_baseline": {"macro_f1": cnn_history[-1]["val_macro_f1"], "micro_f1": cnn_history[-1]["val_micro_f1"]},
    }
    save_json(comparison, os.path.join(cfg["paths"]["results_dir"], "task2_gnn_vs_cnn_comparison.json"))
    print(f"[task2] GNN vs CNN baseline (final-epoch val): {comparison}")

    return encoder, head


def build_task2_cnn_dataloader(cfg: dict, split: str) -> DataLoader:
    fma_cfg = cfg["dataset"]["fma"]
    fma_split_names = {"train": "training", "val": "validation", "test": "test"}
    max_samples = cfg["dataset"].get("max_samples")

    dataset = FMAMelDataset(
        tracks_csv=fma_cfg["tracks_csv"],
        audio_dir=fma_cfg["audio_dir"],
        subset=fma_cfg["subset"],
        split=fma_split_names[split],
        num_genre_classes=cfg["dataset"]["num_genre_classes"],
        sample_rate=cfg["dataset"]["sample_rate"],
        hop_length=cfg["audio_features"]["hop_length"],
        n_mels=cfg["audio_features"]["n_mels"],
        max_samples=max_samples,
        cache_dir=os.path.join(cfg["data"]["processed_dir"], "fma_mel"),
    )
    return DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=(split == "train"))


def train_task2_cnn_baseline(cfg: dict, device: torch.device):
    train_loader = build_task2_cnn_dataloader(cfg, "train")
    val_loader = build_task2_cnn_dataloader(cfg, "val")
    num_classes = cfg["dataset"]["num_genre_classes"]

    model = SimpleCNN(num_classes=num_classes, n_mels=cfg["audio_features"]["n_mels"]).to(device)
    optimizer = make_optimizer(cfg, scratch_modules=[model])
    loss_fn = nn.BCEWithLogitsLoss()

    def forward_eval(batch, dev):
        x, y = batch
        x = x.to(dev)
        return model(x), y

    history = []
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)

        model.eval()
        y_true, y_probs = collect_multilabel_predictions(forward_eval, val_loader, device=device)
        y_pred = single_label_predictions_to_onehot(y_probs)  
        f1_metrics = ev.precision_recall_f1_per_tag(y_true, y_pred)

        print(
            f"[task2-cnn-baseline] epoch {epoch + 1}/{cfg['training']['epochs']} train_loss={train_loss:.4f} "
            f"val_macro_f1={f1_metrics['macro_f1']:.4f} val_micro_f1={f1_metrics['micro_f1']:.4f}"
        )
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_macro_f1": f1_metrics["macro_f1"], "val_micro_f1": f1_metrics["micro_f1"],
        })

    plot_f1_curves(
        history, os.path.join(cfg["paths"]["plots_dir"], "task2_cnn_baseline_f1_curves.png"),
        title="Task 2 baseline: CNN on mel-spectrogram",
    )
    save_json(history, os.path.join(cfg["paths"]["results_dir"], "task2_cnn_baseline_metrics.json"))
    save_checkpoint(
        {"model": model.state_dict()},
        os.path.join(cfg["paths"]["checkpoints_dir"], "task2_cnn_baseline_model.pt"),
    )
    return model, history


# Task 3: GNN-BERT cross-attention fusion 

def build_model_task3(cfg: dict, in_dim: int, num_tags: int, predict_emotion: bool, device: torch.device):
    graph_encoder = GraphEncoder(
        in_dim=in_dim,
        hidden_dim=cfg["model"]["gnn"]["hidden_dim"],
        num_layers=cfg["model"]["gnn"]["num_layers"],
        encoder_type=cfg["model"]["gnn"]["type"],
        dropout=cfg["model"]["gnn"]["dropout"],
    )
    text_encoder = BERTEncoder(
        model_name=cfg["model"]["bert"]["name"],
        freeze_base=cfg["model"]["bert"]["freeze_base"],
    )
    model = GNNBERTFusionModel(
        graph_encoder=graph_encoder,
        text_encoder=text_encoder,
        graph_dim=graph_encoder.hidden_dim,
        text_dim=text_encoder.hidden_dim,
        num_tags=num_tags,
        predict_emotion=predict_emotion,
    ).to(device)
    return model


def run_step_task3(model, batch, tag_loss_fn, emotion_loss_weight: float, device: torch.device):
    graph_batch = batch["graph_batch"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    tag_labels = batch["tag_labels"].to(device)

    out = model(graph_batch, input_ids, attention_mask)
    loss = tag_loss_fn(out["tag_logits"], tag_labels.float())

    if model.predict_emotion and "valence" in batch and "arousal" in batch:
        valence = batch["valence"].to(device)
        arousal = batch["arousal"].to(device)
        emotion_loss = nn.functional.mse_loss(out["valence"], valence.float()) + \
                       nn.functional.mse_loss(out["arousal"], arousal.float())
        loss = loss + emotion_loss_weight * emotion_loss

    return loss, out


def train_task3(cfg: dict):
    device = get_device()
    print(f"[task3] Running on device: {device}")
    train_loader = build_dataloader(cfg, task=3, split="train")
    val_loader = build_dataloader(cfg, task=3, split="val")

    first_batch = next(iter(train_loader))
    in_dim = first_batch["graph_batch"].x.shape[1]
    num_tags = first_batch["tag_labels"].shape[1]  
    predict_emotion = "valence" in first_batch and "arousal" in first_batch
    results_dir, plots_dir, ckpt_dir = cfg["paths"]["results_dir"], cfg["paths"]["plots_dir"], cfg["paths"]["checkpoints_dir"]

    model = build_model_task3(cfg, in_dim, num_tags, predict_emotion, device)

    def forward_main(batch, dev):
        graph_batch = batch["graph_batch"].to(dev)
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch["attention_mask"].to(dev)
        out = model(graph_batch, input_ids, attention_mask)
        return out["tag_logits"], batch["tag_labels"]

    history = run_task3_variant(
        "cross-attention", forward_main, train_loader, val_loader, cfg, device,
        scratch_modules=[model.graph_encoder, model.fusion, model.tag_head] + ([model.emotion_head] if model.predict_emotion else []),
        pretrained_modules=[model.text_encoder],
    )
    plot_f1_curves(
        history, os.path.join(plots_dir, "task3_cross_attention_f1_curves.png"),
        title="Task 3: GNN-BERT cross-attention fusion",
    )
    save_json(history, os.path.join(results_dir, "task3_cross_attention_metrics.json"))
    save_checkpoint(
        {"model": model.state_dict(), "genre_names": getattr(val_loader.dataset, "genre_names", None), "in_dim": in_dim},
        os.path.join(ckpt_dir, "task3_cross_attention_model.pt"),
    )

    bert_encoder = BERTEncoder(model_name=cfg["model"]["bert"]["name"], freeze_base=cfg["model"]["bert"]["freeze_base"]).to(device)
    bert_head = TagClassifierHead(hidden_dim=bert_encoder.hidden_dim, num_tags=num_tags).to(device)

    def forward_bert_only(batch, dev):
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch["attention_mask"].to(dev)
        cls_embedding, _ = bert_encoder(input_ids, attention_mask)
        return bert_head(cls_embedding), batch["tag_labels"]

    hist_bert = run_task3_variant(
        "bert-only", forward_bert_only, train_loader, val_loader, cfg, device,
        scratch_modules=[bert_head], pretrained_modules=[bert_encoder],
    )

    gnn_encoder = GraphEncoder(
        in_dim=in_dim, hidden_dim=cfg["model"]["gnn"]["hidden_dim"], num_layers=cfg["model"]["gnn"]["num_layers"],
        encoder_type=cfg["model"]["gnn"]["type"], dropout=cfg["model"]["gnn"]["dropout"],
    ).to(device)
    gnn_head = GraphTagHead(hidden_dim=gnn_encoder.hidden_dim, num_classes=num_tags).to(device)

    def forward_gnn_only(batch, dev):
        graph_batch = batch["graph_batch"].to(dev)
        _, g = gnn_encoder(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
        return gnn_head(g), batch["tag_labels"]

    hist_gnn = run_task3_variant(
        "gnn-only", forward_gnn_only, train_loader, val_loader, cfg, device,
        scratch_modules=[gnn_encoder, gnn_head],
    )

    early_graph_encoder = GraphEncoder(
        in_dim=in_dim, hidden_dim=cfg["model"]["gnn"]["hidden_dim"], num_layers=cfg["model"]["gnn"]["num_layers"],
        encoder_type=cfg["model"]["gnn"]["type"], dropout=cfg["model"]["gnn"]["dropout"],
    )
    early_text_encoder = BERTEncoder(model_name=cfg["model"]["bert"]["name"], freeze_base=cfg["model"]["bert"]["freeze_base"])
    early_model = EarlyConcatFusionModel(
        early_graph_encoder, early_text_encoder, early_graph_encoder.hidden_dim, early_text_encoder.hidden_dim,
        num_tags, predict_emotion,
    ).to(device)

    def forward_early(batch, dev):
        graph_batch = batch["graph_batch"].to(dev)
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch["attention_mask"].to(dev)
        out = early_model(graph_batch, input_ids, attention_mask)
        return out["tag_logits"], batch["tag_labels"]

    hist_early = run_task3_variant(
        "early-concat", forward_early, train_loader, val_loader, cfg, device,
        scratch_modules=[early_model.graph_encoder, early_model.tag_head] + ([early_model.emotion_head] if early_model.predict_emotion else []),
        pretrained_modules=[early_model.text_encoder],
    )

    ablation_comparison = {
        "bert_only": _final_metrics(hist_bert),
        "gnn_only": _final_metrics(hist_gnn),
        "early_concat": _final_metrics(hist_early),
        "cross_attention": _final_metrics(history),
    }
    save_json(ablation_comparison, os.path.join(results_dir, "task3_ablation_comparison.json"))
    print(f"[task3] ablation comparison (final-epoch val): {ablation_comparison}")

    genre_names = getattr(val_loader.dataset, "genre_names", None)
    generate_tsne_plot(model, val_loader, genre_names, os.path.join(plots_dir, "task3_tsne.png"), device=device, seed=cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["bert"]["name"])
    generate_case_studies(
        model, val_loader, tokenizer, genre_names, os.path.join(results_dir, "task3_case_studies.json"), device=device, n=3,
    )

    return model


def _final_metrics(history: list[dict]) -> dict:
    last = history[-1]
    return {"macro_f1": last["val_macro_f1"], "micro_f1": last["val_micro_f1"], "mean_auc_pr": last["val_mean_auc_pr"]}


def run_task3_variant(
    name: str, forward_fn, train_loader, val_loader, cfg: dict, device: torch.device,
    scratch_modules: list[nn.Module] = (), pretrained_modules: list[nn.Module] = (),
) -> list[dict]:
    modules = list(scratch_modules) + list(pretrained_modules)
    optimizer = make_optimizer(cfg, scratch_modules=scratch_modules, pretrained_modules=pretrained_modules)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(cfg["training"]["epochs"]):
        for m in modules:
            m.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits, labels = forward_fn(batch, device)
            loss = loss_fn(logits, labels.to(device).float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)

        for m in modules:
            m.eval()
        y_true, y_probs = collect_multilabel_predictions(forward_fn, val_loader, device=device)
        y_pred = single_label_predictions_to_onehot(y_probs)
        f1_metrics = ev.precision_recall_f1_per_tag(y_true, y_pred)
        auc_metrics = ev.auc_pr_per_tag(y_true, y_probs)

        print(
            f"[task3-{name}] epoch {epoch + 1}/{cfg['training']['epochs']} train_loss={train_loss:.4f} "
            f"val_macro_f1={f1_metrics['macro_f1']:.4f} val_mean_auc_pr={auc_metrics['mean_auc_pr']:.4f}"
        )
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_macro_f1": f1_metrics["macro_f1"], "val_micro_f1": f1_metrics["micro_f1"],
            "val_mean_auc_pr": auc_metrics["mean_auc_pr"],
        })
    return history


@torch.no_grad()
def generate_tsne_plot(model, val_loader, genre_names, out_path: str, device: torch.device, seed: int = 42):
    model.eval()
    zs, labels = [], []
    for batch in val_loader:
        graph_batch = batch["graph_batch"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(graph_batch, input_ids, attention_mask)
        zs.append(out["z"].cpu().numpy())
        labels.append(batch["tag_labels"].argmax(dim=1).cpu().numpy())
    zs = np.concatenate(zs)
    labels = np.concatenate(labels)

    if len(zs) < 5:
        print(f"[t-sne] only {len(zs)} validation examples -- too few for a meaningful t-SNE, skipping")
        return

    perplexity = min(30, max(2, len(zs) // 3))
    coords = TSNE(n_components=2, perplexity=perplexity, random_state=seed).fit_transform(zs)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=25)
    if genre_names:
        handles, _ = scatter.legend_elements(num=len(set(labels.tolist())))
        present_names = [genre_names[i] for i in sorted(set(labels.tolist()))]
        plt.legend(handles, present_names, title="Genre", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.title("t-SNE of fused representation z (colored by genre)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[t-sne] saved to {out_path}")


@torch.no_grad()
def generate_case_studies(model, val_loader, tokenizer, genre_names, out_path: str, device: torch.device, n: int = 3):
    model.eval()
    dataset = val_loader.dataset
    cases = []
    for i in range(min(n, len(dataset))):
        item = dataset[i]
        batch = val_loader.collate_fn([item])
        graph_batch = batch["graph_batch"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        out = model(graph_batch, input_ids, attention_mask, return_attention=True)

        probs = torch.sigmoid(out["tag_logits"])[0]
        pred_idx = probs.argmax().item()
        true_idx = item["tag_labels"].argmax().item()

        tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][0].tolist())
        attn = out["attention_weights"][0].tolist()
        token_attention = sorted(
            [(tok, round(w, 4)) for tok, w in zip(tokens, attn) if tok not in tokenizer.all_special_tokens],
            key=lambda x: -x[1],
        )

        graph = item["graph"]
        edge_index = graph.edge_index
        edge_weights = graph.edge_attr.squeeze(-1).tolist() if graph.edge_attr is not None else [1.0] * edge_index.shape[1]
        top_edges = sorted(
            zip(edge_index[0].tolist(), edge_index[1].tolist(), edge_weights), key=lambda e: -e[2]
        )[:3]

        cases.append({
            "input_text": item["text"],
            "true_genre": genre_names[true_idx] if genre_names else true_idx,
            "predicted_genre": genre_names[pred_idx] if genre_names else pred_idx,
            "top_attended_tokens": token_attention[:5],
            "graph_num_nodes": graph.x.shape[0],
            "graph_num_edges": graph.edge_index.shape[1],
            "graph_top_edges_by_weight": top_edges,
        })

    save_json(cases, out_path)
    print(f"[case studies] saved to {out_path}")
    
    


TASK_FNS = {
    1: train_task1,
    2: train_task2,
    3: train_task3
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Quick-test: cap each split to this many examples (e.g. 50) instead of the full dataset.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override config.yaml's training.epochs (e.g. --epochs 2 for a quick test).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_samples is not None:
        cfg["dataset"]["max_samples"] = args.max_samples
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    dataset_name = resolve_dataset_name(cfg, args.task)
    print(f"[train] task={args.task} dataset={dataset_name} max_samples={args.max_samples} epochs={cfg['training']['epochs']}")
    TASK_FNS[args.task](cfg)


if __name__ == "__main__":
    main()