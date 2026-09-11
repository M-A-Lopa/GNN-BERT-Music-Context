
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score


def precision_recall_f1_per_tag(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
   
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    macro_f1 = float(np.mean(f1))
    _, _, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    return {
        "per_tag": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
        },
        "macro_f1": macro_f1,
        "micro_f1": float(micro_f1),
    }


def auc_pr_per_tag(y_true: np.ndarray, y_scores: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if y_true.shape != y_scores.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_scores shape {y_scores.shape}")

    num_tags = y_true.shape[1]
    per_tag_auc_pr = []
    for k in range(num_tags):
        if y_true[:, k].sum() == 0:
            per_tag_auc_pr.append(float("nan"))
        else:
            per_tag_auc_pr.append(float(average_precision_score(y_true[:, k], y_scores[:, k])))

    valid = [v for v in per_tag_auc_pr if not np.isnan(v)]
    mean_auc_pr = float(np.mean(valid)) if valid else float("nan")

    return {"per_tag_auc_pr": per_tag_auc_pr, "mean_auc_pr": mean_auc_pr}


def emotion_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("emotion_regression_metrics received empty arrays.")

    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"mae": mae, "r2": r2}


def graph_coherence_score(node_embeddings: np.ndarray, edges: np.ndarray, threshold: float) -> float:
  
    node_embeddings = np.asarray(node_embeddings)
    edges = np.asarray(edges)
    if edges.size == 0:
        raise ValueError("graph_coherence_score received an empty edge list.")

    norms = np.linalg.norm(node_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    normalized = node_embeddings / norms

    src, dst = edges[:, 0], edges[:, 1]
    sims = np.sum(normalized[src] * normalized[dst], axis=1)  

    return float(np.mean(sims > threshold))
