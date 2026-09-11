

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class BERTEncoder(nn.Module):
    

    def __init__(self, model_name: str = "bert-base-uncased", freeze_base: bool = False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.bert.config.hidden_size

        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state          
        cls_embedding = token_embeddings[:, 0, :]              
        return cls_embedding, token_embeddings


class TagClassifierHead(nn.Module):

    def __init__(self, hidden_dim: int, num_tags: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_tags)

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        return self.linear(cls_embedding)
