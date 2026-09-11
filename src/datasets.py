

from __future__ import annotations

import ast
import os
import random

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from audio_features import load_and_resample, extract_chroma, extract_log_mel, normalize_track, segment_track
from graph_builder import build_segment_graph

MASK_RATIO = 0.5  


def load_top_k_tags(annotations_csv: str, top_k: int = 50) -> list[str]:
    
    df = pd.read_csv(annotations_csv, sep="\t")
    tag_cols = [c for c in df.columns if c not in ("clip_id", "mp3_path")]
    counts = df[tag_cols].sum(axis=0).sort_values(ascending=False)
    return counts.index[:top_k].tolist()


class MagnaTagATuneDataset(Dataset):
   

    def __init__(
        self,
        annotations_csv: str,
        split_ids_tsv: str,
        top_k_tags: int = 50,
        mask_ratio: float = MASK_RATIO,
        seed: int = 42,
        max_samples: int | None = None,
    ):
        df = pd.read_csv(annotations_csv, sep="\t")
        self.tag_names = load_top_k_tags(annotations_csv, top_k_tags)

        
        df = df[df[self.tag_names].sum(axis=1) > 0]

        split_ids = pd.read_csv(split_ids_tsv, sep="\t", header=None)[0].tolist()
        df = df[df["clip_id"].isin(split_ids)].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(
                f"No clips found for split {split_ids_tsv} after filtering against "
                f"{annotations_csv} -- check the paths and that clip_id values line up."
            )
        if max_samples is not None:
            df = df.head(max_samples)  

        self.clip_ids = df["clip_id"].tolist()
        self.labels = df[self.tag_names].to_numpy().astype("float32")  
        self.mp3_paths = df["mp3_path"].tolist()
        self.mask_ratio = mask_ratio
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.clip_ids)

    def __getitem__(self, idx: int) -> dict:
        label_vec = self.labels[idx]
        present_tags = [self.tag_names[i] for i, v in enumerate(label_vec) if v > 0]

        n_visible = max(1, round(len(present_tags) * self.mask_ratio))
        visible_tags = self._rng.sample(present_tags, min(n_visible, len(present_tags)))
        text = ", ".join(visible_tags) if visible_tags else "no tags"

        return {"text": text, "labels": torch.tensor(label_vec, dtype=torch.float32)}


def collate_fn(batch: list[dict], tokenizer, max_length: int = 256) -> dict:
    texts = [item["text"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])

    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


# FMA 

_FMA_SUBSET_RANK = {"small": 1, "medium": 2, "large": 3}


def fma_track_path(track_id: int, audio_dir: str) -> str:
    tid = f"{track_id:06d}"
    return os.path.join(audio_dir, tid[:3], f"{tid}.mp3")


def _load_fma_tracks(
    tracks_csv: str, subset: str, split: str, num_genre_classes: int
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])

    subset_rank = df[("set", "subset")].map(_FMA_SUBSET_RANK)
    df_subset = df[subset_rank <= _FMA_SUBSET_RANK[subset]]
    df_subset = df_subset[df_subset[("track", "genre_top")].notna()]

    genre_names = df_subset[("track", "genre_top")].value_counts().index[:num_genre_classes].tolist()

    df = df_subset[df_subset[("set", "split")] == split]
    df = df[df[("track", "genre_top")].isin(genre_names)]

    if len(df) == 0:
        raise ValueError(
            f"No FMA tracks left after filtering (subset<={subset!r}, split={split!r}) "
            f"against {tracks_csv} -- check the path and subset/split values."
        )
    return df, genre_names


def _track_to_graph(
    track_id: int,
    audio_dir: str,
    sample_rate: int,
    hop_length: int,
    segment_seconds: float,
    similarity_threshold: float,
    feature_type: str,
    n_mels: int,
    n_chroma: int,
    cache_dir: str | None = None,
):
    
    if cache_dir is not None:
        cache_path = os.path.join(cache_dir, f"{track_id}.pt")
        if os.path.exists(cache_path):
            try:
                return torch.load(cache_path, weights_only=False)
            except Exception as e:
                print(f"[Warning] Cache file for track {track_id} unreadable ({e}), recomputing.")

    path = fma_track_path(track_id, audio_dir)
    try:
        y = load_and_resample(path, sample_rate=sample_rate)

        if feature_type == "chroma":
            feat = extract_chroma(y, sample_rate=sample_rate, n_chroma=n_chroma, hop_length=hop_length)
        else:
            feat = extract_log_mel(y, sample_rate=sample_rate, n_mels=n_mels, hop_length=hop_length)
        feat = normalize_track(feat)

        segments = segment_track(
            feat, sample_rate=sample_rate, hop_length=hop_length, mode="fixed", segment_seconds=segment_seconds
        )
        if len(segments) < 2:
            segments = segments * 2

        pooled = [seg.mean(axis=1) for seg in segments]
        graph = build_segment_graph(pooled, similarity_threshold=similarity_threshold)

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(graph, os.path.join(cache_dir, f"{track_id}.pt"))

        return graph
    except Exception as e:
        print(f"[Warning] Skipping corrupted audio track {track_id} ({path}): {e}")
        return None


class FMAMelDataset(Dataset):

    def __init__(
        self,
        tracks_csv: str,
        audio_dir: str,
        subset: str = "medium",
        split: str = "training",
        num_genre_classes: int = 16,
        sample_rate: int = 22050,
        hop_length: int = 512,
        n_mels: int = 128,
        fixed_frames: int = 1300,  
        max_samples: int | None = None,
        cache_dir: str | None = None,
    ):
        df, self.genre_names = _load_fma_tracks(tracks_csv, subset, split, num_genre_classes)
        if max_samples is not None:
            df = df.head(max_samples) 
        genre_to_idx = {g: i for i, g in enumerate(self.genre_names)}

        self.track_ids = df.index.tolist()
        self.genre_idx = [genre_to_idx[g] for g in df[("track", "genre_top")]]
        self.audio_dir = audio_dir
        self.num_genre_classes = num_genre_classes
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fixed_frames = fixed_frames
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.track_ids)

    def _load_mel(self, idx: int):
        track_id = self.track_ids[idx]
        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, f"{track_id}.pt")
            if os.path.exists(cache_path):
                try:
                    return torch.load(cache_path, weights_only=False)
                except Exception as e:
                    print(f"[Warning] Cache file for track {track_id} unreadable ({e}), recomputing.")

        path = fma_track_path(track_id, self.audio_dir)
        try:
            y = load_and_resample(path, sample_rate=self.sample_rate)
            mel = extract_log_mel(y, sample_rate=self.sample_rate, n_mels=self.n_mels, hop_length=self.hop_length)
            mel = normalize_track(mel)

            t = mel.shape[1]
            if t >= self.fixed_frames:
                mel = mel[:, : self.fixed_frames]
            else:
                mel = np.pad(mel, ((0, 0), (0, self.fixed_frames - t)), mode="constant")
            mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  

            if self.cache_dir is not None:
                os.makedirs(self.cache_dir, exist_ok=True)
                torch.save(mel_tensor, os.path.join(self.cache_dir, f"{track_id}.pt"))

            return mel_tensor
        except Exception as e:
            print(f"[Warning] Skipping corrupted audio track {track_id} ({path}): {e}")
            return None

    def __getitem__(self, idx: int):
        n = len(self)
        for offset in range(n):  
            candidate = (idx + offset) % n
            mel_tensor = self._load_mel(candidate)
            if mel_tensor is not None:
                label = torch.zeros(self.num_genre_classes, dtype=torch.float32)
                label[self.genre_idx[candidate]] = 1.0
                return mel_tensor, label
        raise RuntimeError("FMAMelDataset: every track in this split failed to load.")


class FMAGraphDataset(Dataset):

    def __init__(
        self,
        tracks_csv: str,
        audio_dir: str,
        subset: str = "medium",
        split: str = "training",
        num_genre_classes: int = 16,
        sample_rate: int = 22050,
        hop_length: int = 512,
        segment_seconds: float = 7.5,
        similarity_threshold: float = 0.8,
        feature_type: str = "chroma",
        n_mels: int = 128,
        n_chroma: int = 12,
        max_samples: int | None = None,
        cache_dir: str | None = None,
    ):
        df, self.genre_names = _load_fma_tracks(tracks_csv, subset, split, num_genre_classes)
        if max_samples is not None:
            df = df.head(max_samples)  
        genre_to_idx = {g: i for i, g in enumerate(self.genre_names)}

        self.track_ids = df.index.tolist()
        self.genre_idx = [genre_to_idx[g] for g in df[("track", "genre_top")]]
        self.audio_dir = audio_dir
        self.num_genre_classes = num_genre_classes
        self._audio_cfg = dict(
            sample_rate=sample_rate, hop_length=hop_length, segment_seconds=segment_seconds,
            similarity_threshold=similarity_threshold, feature_type=feature_type,
            n_mels=n_mels, n_chroma=n_chroma, cache_dir=cache_dir,
        )

    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, idx: int):
        n = len(self)
        for offset in range(n):  
            candidate = (idx + offset) % n
            graph = _track_to_graph(self.track_ids[candidate], self.audio_dir, **self._audio_cfg)
            if graph is not None:
                label = torch.zeros(self.num_genre_classes, dtype=torch.float32)
                label[self.genre_idx[candidate]] = 1.0
                graph.y = label.unsqueeze(0)
                return graph
        raise RuntimeError("FMAGraphDataset: every track in this split failed to load.")


class FMAGraphTextDataset(Dataset):

    def __init__(
        self,
        tracks_csv: str,
        audio_dir: str,
        raw_tracks_csv: str,
        subset: str = "medium",
        split: str = "training",
        num_genre_classes: int = 16,
        sample_rate: int = 22050,
        hop_length: int = 512,
        segment_seconds: float = 7.5,
        similarity_threshold: float = 0.8,
        feature_type: str = "chroma",
        n_mels: int = 128,
        n_chroma: int = 12,
        max_samples: int | None = None,
        cache_dir: str | None = None,
    ):
        df, self.genre_names = _load_fma_tracks(tracks_csv, subset, split, num_genre_classes)
        if max_samples is not None:
            df = df.head(max_samples)  
        genre_to_idx = {g: i for i, g in enumerate(self.genre_names)}

        raw = pd.read_csv(raw_tracks_csv, index_col="track_id", usecols=["track_id", "tags"])
        self.tags_by_id = raw["tags"].to_dict()

        self.track_ids = df.index.tolist()
        self.genre_idx = [genre_to_idx[g] for g in df[("track", "genre_top")]]
        self.audio_dir = audio_dir
        self.num_genre_classes = num_genre_classes
        self._audio_cfg = dict(
            sample_rate=sample_rate, hop_length=hop_length, segment_seconds=segment_seconds,
            similarity_threshold=similarity_threshold, feature_type=feature_type,
            n_mels=n_mels, n_chroma=n_chroma, cache_dir=cache_dir,
        )

    def __len__(self) -> int:
        return len(self.track_ids)

    def _text_for(self, track_id: int) -> str:
        raw_tags = self.tags_by_id.get(track_id)
        if not raw_tags or raw_tags in ("[]", "nan"):
            return "no tags"
        try:
            tags = ast.literal_eval(raw_tags) if isinstance(raw_tags, str) else raw_tags
        except (ValueError, SyntaxError):
            return "no tags"
        return ", ".join(tags) if tags else "no tags"

    def __getitem__(self, idx: int) -> dict:
        n = len(self)
        for offset in range(n): 
            candidate = (idx + offset) % n
            track_id = self.track_ids[candidate]
            graph = _track_to_graph(track_id, self.audio_dir, **self._audio_cfg)
            if graph is not None:
                label = torch.zeros(self.num_genre_classes, dtype=torch.float32)
                label[self.genre_idx[candidate]] = 1.0
                return {"graph": graph, "text": self._text_for(track_id), "tag_labels": label, "track_id": track_id}
        raise RuntimeError("FMAGraphTextDataset: every track in this split failed to load.")


def fma_task3_collate_fn(batch: list[dict], tokenizer, max_length: int = 256) -> dict:
    graph_batch = Batch.from_data_list([item["graph"] for item in batch])
    texts = [item["text"] for item in batch]
    tag_labels = torch.stack([item["tag_labels"] for item in batch])
    track_ids = [item["track_id"] for item in batch]

    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    return {
        "graph_batch": graph_batch,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "tag_labels": tag_labels,
        "track_ids": track_ids,
    }