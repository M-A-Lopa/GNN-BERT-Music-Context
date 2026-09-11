
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

try:
    from torch_geometric.data import Data
except ImportError:  
    Data = None


def _require_pyg():
    if Data is None:
        raise ImportError(
            "torch_geometric is required for graph construction. "
            "Install it via `pip install torch_geometric` (see requirements.txt)."
        )


def build_chord_transition_graph(chord_sequence: list[str], min_count: int = 1) -> "Data":
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

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() 
    edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(1)    
    x = torch.eye(num_nodes, dtype=torch.float)                       

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.chord_vocab = unique_chords 
    return data


def build_segment_graph(
    segment_features: list[np.ndarray],
    similarity_threshold: float = 0.8,
) -> "Data":
    _require_pyg()

    if len(segment_features) < 2:
        raise ValueError("segment_features must have at least 2 segments to form a graph.")

    features = np.stack([np.asarray(f).reshape(-1) for f in segment_features]) 
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
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() 
    edge_attr = torch.tensor(
        [sim[i, j] for i, j in edges], dtype=torch.float
    ).unsqueeze(1)  
    x = torch.tensor(features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms) 
    normalized = features / norms
    return normalized @ normalized.T