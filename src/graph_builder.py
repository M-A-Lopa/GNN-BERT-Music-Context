
from __future__ import annotations

from collections import Counter

import pandas as pd
import numpy as np
import torch

try:
    from torch_geometric.data import Data
except ImportError:  # torch-geometric not installed yet in this environment
    Data = None


def _require_pyg():
    if Data is None:
        raise ImportError(
            "torch_geometric is required for graph construction. "
            "Install it via `pip install torch_geometric` (see requirements.txt)."
        )


def build_chord_transition_graph(chord_sequence: list[str], min_count: int = 1) -> "Data":
    """
    Build a chord-transition graph from a sequence of chord labels.

    Nodes = unique chords in `chord_sequence`.
    Edge (i, j) weight = number of observed transitions chord_i -> chord_j,
    edges with weight < min_count are dropped.

    Node features (x) are one-hot vectors over the unique-chord vocabulary —
    a simple, deterministic placeholder; swap for learned chord embeddings
    later if desired.
    """
    _require_pyg()

    if len(chord_sequence) < 2:
        raise ValueError("chord_sequence must have at least 2 chords to form a transition.")

    unique_chords = sorted(set(chord_sequence))
    chord_to_idx = {c: i for i, c in enumerate(unique_chords)}
    num_nodes = len(unique_chords)

    transition_counts = Counter(
        (chord_to_idx[chord_sequence[i]], chord_to_idx[chord_sequence[i + 1]])
        for i in range(len(chord_sequence) - 1)
    )

    edges, weights = [], []
    for (src, dst), count in transition_counts.items():
        if count >= min_count:
            edges.append((src, dst))
            weights.append(count)

    if len(edges) == 0:
        raise ValueError(
            f"No transitions survived min_count={min_count}; lower min_count or "
            "check chord_sequence."
        )

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, num_edges)
    edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(1)     # (num_edges, 1)
    x = torch.eye(num_nodes, dtype=torch.float)                          # one-hot node features

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.chord_vocab = unique_chords  # keep the mapping around for inspection/debugging
    return data


def build_segment_graph(
    segment_features: list[np.ndarray],
    similarity_threshold: float = 0.8,
) -> "Data":
    """
    Build a segment graph from a list of per-segment feature vectors
    (e.g. mean-pooled MFCC/chroma per segment).

    Edges:
      - temporal adjacency: segment i <-> segment i+1
      - similarity: segment i <-> segment j if cosine_sim(i, j) > similarity_threshold

    Both edge types are combined and de-duplicated; edges are undirected
    (both (i, j) and (j, i) are added).
    """
    _require_pyg()

    if len(segment_features) < 2:
        raise ValueError("segment_features must have at least 2 segments to form a graph.")

    features = np.stack([np.asarray(f).reshape(-1) for f in segment_features])  # (N, D)
    num_nodes = features.shape[0]

    sim = cosine_similarity_matrix(features)

    edge_set = set()
    for i in range(num_nodes - 1):
        edge_set.add((i, i + 1))
        edge_set.add((i + 1, i))

    above_thresh = np.argwhere(sim > similarity_threshold)
    for i, j in above_thresh:
        if i != j:
            edge_set.add((int(i), int(j)))

    edges = sorted(edge_set)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, num_edges)
    edge_attr = torch.tensor(
        [sim[i, j] for i, j in edges], dtype=torch.float
    ).unsqueeze(1)  # cosine similarity as edge weight (temporal-only edges use actual sim too)
    x = torch.tensor(features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity for an (N, D) feature matrix -> (N, N)."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)  # avoid div-by-zero for all-zero rows
    normalized = features / norms
    return normalized @ normalized.T