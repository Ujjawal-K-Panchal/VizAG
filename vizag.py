"""
Title: VizAG.

Date: May 27, 2024

Author: Ujjawal K. Panchal & Ajinkya Chaudhari & Isha S. Joglekar
"""
import torch.nn as nn

import projconfig
import baseclip, retrieval, generation

class VizAG(nn.Module):
    kernel_prompt = """
    Based on the given documents, answer the user's query.

    Documents:
    {docs}

    Query:
    {query}

    Answer:
    """
    def __init__(
        self, images, clipproc,
        clipmodel, embedder, gen_tok,
        gen_model, k: int = projconfig.k
    ):
        self.images = images
        self.clipproc, self.clipmodel = clipproc, clipmodel
        self.embedder = embedder
        self.generator = generator
        self.docs = []
        self.docembs = []
        return

    def setup(self):
        """
        Desc:
            Factory to preprocess images and put them in `doc2img` (a string to image map).
        """
        for image in self.images:
            #1. caption it.
            cap = baseclip.img2cap(
                    self.clipproc,
                    self.clipmodel,
                    image,
                    device = self.clipmodel.device
            )
            self.docs.append(cap)
            #2. embed it.
            emb = retrieval.embed_strings(
                    self.embedder,
                    cap,
                    layer_index = projconfig.layer_index,
                    embedding_size = projconfig.embedding_size,
                    device = self.embedder.device
            )
            self.docembs.append(emb)
        return

    def generate(self, query):
        """
        Desc:
            Use the image captions you have to answer user's query.
        """
        with torch.no_grad():
            #1. find topk docs.
            top_k_docs = find_topk_embs(
                query = query,
                model = self.embedder,
                docs = self.docs,
                embs = self.docembs,
                k = self.k,
                layer_index: int = projconfig.layer_index,
                embedding_size: int = projconfig.embedding_size,
            ) 
            docstring = "\n".join(top_k_docs)
            #2. append topk docs to query to answer the question.
            augmented_query =  VizAG.kernel_prompt.format(docs = docstring, query = query)
            #3. generate message using generator.
            genmsg =  generation.get_reply(augmented_query, self.generator)
        return genmsg

if __name__ == "__main__":
    #TODO:
    ...
