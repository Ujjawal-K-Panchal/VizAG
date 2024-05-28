"""
Title: Test Retriever portion.

Date: May 27, 2024

Author: Ujjawal K. Panchal & Ajinkya Chaudhari & Isha S. Joglekar
"""
import torch
from typing import Iterable
from angle_emb import AnglE
from sentence_transformers.util import cos_sim

import projconfig

model = AnglE.from_pretrained(
    "mixedbread-ai/mxbai-embed-2d-large-v1",
    pooling_strategy='cls',
    cache_dir = projconfig.modelstore
).cuda()


def embed_strings(model, strings: str, layer_index: int = projconfig.layer_index, embedding_size: int = projconfig.embedding_size):
    return model.encode(strings, layer_index = layer_index, embedding_size = embedding_size)


def find_topk_embs(
    query: str,
    docs: Iterable,
    embs: Iterable,
    k: int = 2,
    layer_index: int = projconfig.layer_index,
    embedding_size: int = projconfig.embedding_size,
):
    """
    Desc:
        Caption 2 Similarity.
    """
    q_emb = embed_strings(model, [query])[0]
    sims = cos_sim(q_emb, embs)
    top_k_indices = torch.argsort(-1*sims)[0]
    return [docs[i] for i in top_k_indices.tolist()][:k]


if __name__ == "__main__":
    #1. embeddings.
    docs = [
        'Who is german and likes bread?',
        'German breads are the best in the world!',
        'French breads are better than German breads.',
        'Everybody in Germany.',
        'Dogs like bones.',
    ]
    embeddings = embed_strings(model, docs)
    #2. embedding similarity.
    topdocs = find_topk_embs("Germany has good breads", docs, embeddings)
    print(topdocs)