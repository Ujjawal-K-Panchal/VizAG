"""
Title: VizAG.

Date: May 27, 2024

Author: Ujjawal K. Panchal & Ajinkya Chaudhari & Isha S. Joglekar
"""
import imageio
from PIL import Image

import torch
import torch.nn as nn

import projconfig
import baseclip, retrieval, generation

class VizAG(nn.Module):
    kernel_prompt = """
    Based on the given descriptions of images, answer the user's query in short.

    Image descriptions:
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
        super().__init__()
        self.images = images
        self.clipproc = clipproc
        self.clipmodel = clipmodel
        self.k = k
        self.embedder = embedder
        self.gen_tok = gen_tok
        self.gen_model = gen_model
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
        #2. embed all docs.
        self.docembs = retrieval.embed_strings(
                    self.embedder,
                    self.docs,
                    layer_index = projconfig.layer_index,
                    embedding_size = projconfig.embedding_size,
                    device = self.embedder.device
        )
        return

    def generate(self, query):
        """
        Desc:
            Use the image captions you have to answer user's query.
        """
        with torch.no_grad():
            #1. find topk docs.
            top_k_docs = retrieval.find_topk_embs(
                query = query,
                model = self.embedder,
                docs = self.docs,
                embs = self.docembs,
                k = self.k,
            ) 
            docstring = "\n".join(top_k_docs)
            #2. append topk docs to query to answer the question.
            augmented_query =  VizAG.kernel_prompt.format(docs = docstring, query = query)
            #3. generate message using generator.
            genmsg =  generation.get_reply(self.gen_model, self.gen_tok, augmented_query)
        return genmsg

if __name__ == "__main__":
    #1. Extra imports.
    from transformers import AutoProcessor, AutoModelForCausalLM
    from angle_emb import AnglE
    #2. get clip stuff.
    clipproc = AutoProcessor.from_pretrained(projconfig.clip_model_name, cache_dir = projconfig.modelstore)
    clipmodel = AutoModelForCausalLM.from_pretrained(projconfig.clip_model_name, cache_dir = projconfig.modelstore).to(projconfig.device)
    #3. get retriever.
    retriever = AnglE.from_pretrained(
        projconfig.emb_model_name,
        pooling_strategy=projconfig.emb_pooling_strategy,
        cache_dir = projconfig.modelstore
    ).to(projconfig.device)
    #4. get generator.
    gen_tok = generation.get_tokenizer(projconfig.llm_name)
    gen_model = generation.get_model(projconfig.llm_name)
    #5. images.
    images = [
        "https://static.toiimg.com/photo/msid-61220500,width-96,height-65.cms",
        "https://lp-cms-production.imgix.net/2021-01/GettyRF_450207051.jpg",
        "https://cdn.britannica.com/74/114874-050-6E04C88C/North-Face-Mount-Everest-Tibet-Autonomous-Region.jpg",
        "https://i.natgeofe.com/n/28fd84d5-ebb6-45c8-a758-17d0089a0464/TT_report_ST0058057copy-Enhanced_HR_4x3.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Swiss_Alps.jpg/1200px-Swiss_Alps.jpg"
    ]
    images = [Image.fromarray(imageio.imread(url)) for url in images]
    #6. make vizag.
    vizag = VizAG(
        images, #images.
        clipproc, clipmodel, #clip stuff.
        retriever, #retriever.
        gen_tok, gen_model, #generator.
    )
    vizag.setup() #factory to setup.
    #7. Make query.
    reply = vizag.generate("How many images have mountains with snow on them?")
    print(f"{reply=}")
    print(f"{vizag.docs=}")
    
