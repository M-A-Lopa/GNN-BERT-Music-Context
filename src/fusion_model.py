

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):

    def __init__(self, graph_dim: int, text_dim: int, attn_dim: int):
        super().__init__()
        self.attn_dim = attn_dim
        self.w_q = nn.Linear(graph_dim, attn_dim, bias=False)
        self.w_k = nn.Linear(text_dim, attn_dim, bias=False)

    def forward(
        self, g: torch.Tensor, h_text: torch.Tensor, attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        q = self.w_q(g).unsqueeze(1)          
        k = self.w_k(h_text)                   

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.attn_dim)  

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).bool()  
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)        
        attended = torch.matmul(attn_weights, h_text).squeeze(1)  

        z = torch.cat([g, attended], dim=-1)             
        if return_attention:
            return z, attn_weights.squeeze(1)  
        return z


class GNNBERTFusionModel(nn.Module):

    def __init__(
        self,
        graph_encoder: nn.Module,
        text_encoder: nn.Module,
        graph_dim: int,
        text_dim: int,
        num_tags: int,
        predict_emotion: bool = False,
        attn_dim: int | None = None,
    ):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.text_encoder = text_encoder
        self.predict_emotion = predict_emotion

        attn_dim = attn_dim or graph_dim
        self.fusion = CrossAttentionFusion(graph_dim, text_dim, attn_dim)

        fused_dim = graph_dim + text_dim
        self.tag_head = nn.Linear(fused_dim, num_tags)
        self.emotion_head = nn.Linear(fused_dim, 2) if predict_emotion else None  

    def forward(
        self, graph_batch, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        return_attention: bool = False,
    ):
        _, g = self.graph_encoder(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
        _, h_text = self.text_encoder(input_ids, attention_mask)

        if return_attention:
            z, attn_weights = self.fusion(g, h_text, attention_mask, return_attention=True)
        else:
            z = self.fusion(g, h_text, attention_mask)
        tag_logits = self.tag_head(z)

        valence, arousal = None, None
        if self.predict_emotion:
            emotion = self.emotion_head(z)
            valence, arousal = emotion[:, 0], emotion[:, 1]

        result = {"tag_logits": tag_logits, "valence": valence, "arousal": arousal, "z": z}
        if return_attention:
            result["attention_weights"] = attn_weights
        return result


class EarlyConcatFusionModel(nn.Module):

    def __init__(
        self,
        graph_encoder: nn.Module,
        text_encoder: nn.Module,
        graph_dim: int,
        text_dim: int,
        num_tags: int,
        predict_emotion: bool = False,
    ):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.text_encoder = text_encoder
        self.predict_emotion = predict_emotion

        fused_dim = graph_dim + text_dim
        self.tag_head = nn.Linear(fused_dim, num_tags)
        self.emotion_head = nn.Linear(fused_dim, 2) if predict_emotion else None

    def forward(self, graph_batch, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        _, g = self.graph_encoder(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
        cls_embedding, _ = self.text_encoder(input_ids, attention_mask)

        z = torch.cat([g, cls_embedding], dim=-1)  
        tag_logits = self.tag_head(z)

        valence, arousal = None, None
        if self.predict_emotion:
            emotion = self.emotion_head(z)
            valence, arousal = emotion[:, 0], emotion[:, 1]

        return {"tag_logits": tag_logits, "valence": valence, "arousal": arousal, "z": z}