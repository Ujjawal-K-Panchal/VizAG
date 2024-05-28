"""
Title: Test Retriever portion.

Date: May 27, 2024

Author: Ujjawal K. Panchal & Ajinkya Chaudhari & Isha S. Joglekar
"""
from angle_emb import AnglE
from sentence_transformers.util import cos_sim

import projconfig


model = AnglE.from_pretrained(
    "mixedbread-ai/mxbai-embed-2d-large-v1",
    pooling_strategy='cls',
    cache_dir = projconfig.modelstore
).cuda()

# it is recommended to set layers from 20 to 24.
layer_index = 22  # 1d: layer
embedding_size = 768  # 2d: embedding size

if __name__ == "__main__":
    # 3. embeddings.
    embeddings = model.encode([
        'Who is german and likes bread?',
        'Everybody in Germany.'
    ], layer_index=layer_index, embedding_size=embedding_size)
    #4. embedding similarity.
    similarities = cos_sim(embeddings[0], embeddings[1:])
    print('similarities:', similarities)