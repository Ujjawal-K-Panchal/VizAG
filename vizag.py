"""
Title: VizAG.

Date: May 27, 2024

Author: Ujjawal K. Panchal & Ajinkya Chaudhari & Isha S. Joglekar
"""
import warnings

import argparse, os, imageio
from typing import Callable, Optional
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

import projconfig
import baseclip, sforceclip #clip models.
import retrieval, generation


warnings.filterwarnings('ignore')

class VizAG(nn.Module):
    kernel_prompt = """
    Based on the given batch of descriptions of images, answer the user's query in short.

    Image descriptions:
    {docs}

    Query:
    {query}

    Answer:
    """

    kernel_combine_prompt = """
    Given is a user's query. Look at answers for isolated image batches, add them up to form an aggregated reply to user's query.

    Eg.
    Reply 1: 10 dogs.
    Reply 2: 1 dog.
    Reply 3: no dogs.
    Final answer: 11 dogs.

    You see how it didnot give the most frequent answer, but it added the number of dogs up from different replies i.e. 10 from reply 1
    and 1 from reply 2 to to give a final answer of 11. Do the same for the given query below.


    Query:
    {query}

    Intermediate answers:
    {docs}

    Final answer: 
    """

    def __init__(
        self, images, clipproc,
        clipmodel, embedder, gen_tok,
        gen_model, k: int = projconfig.k,
        bs: int = projconfig.batchsize,
        db_device: str | torch.device = projconfig.db_device,
        img2cap: Callable = baseclip.img2cap, #defaulting to CLIP's img2cap.
        img2capkwargs: Optional[dict] = None,

    ):
        super().__init__()
        self.images = images
        self.clipproc = clipproc
        self.clipmodel = clipmodel
        self.k = k
        self.bs = bs
        self.embedder = embedder.to(db_device) #embedder goes into database device.
        self.gen_tok = gen_tok
        self.gen_model = gen_model
        self.docs = []
        self.docembs = []
        self.db_device = db_device
        self.img2cap = img2cap
        self.img2capkwargs = img2capkwargs
        return
    
    def save_snapshot(self, shotpath: Path | str):
        """
        Desc:
            Setting up VizAG takes some time for large datasets. Hence, prudent to store it away for future use after first setup.
        """
        torch.save([self.docs, self.docembs], shotpath)
        return

    def setup_snapshot(self, shotpath: Path | str):
        """
        Desc:
            Factory to load dataset from snapshot. This is to avoid high processing time for captioning images & embedding captions. :/
        """
        with open(shotpath, "rb") as shotfile:
            self.docs, self.docembs = torch.load(shotfile)
        if not isinstance(self.docembs, torch.Tensor):
            self.docembs = torch.stack(self.docembs)
        self.docembs.to(self.db_device)
        return

    def setup(self):
        """
        Desc:
            Factory to preprocess images and put them in `doc2img` (a string to image map).
        """
        with tqdm(range(0, len(self.images), self.bs), unit = "batches") as batchiter:
            for i in batchiter:
                #1. caption image batch.
                cap = self.img2cap(
                        processor = self.clipproc,
                        model = self.clipmodel,
                        img = self.images[i:i + self.bs], #`self.bs` number of images captioned at once.
                        device = self.clipmodel.device,
                        **self.img2capkwargs #extra keyword arguments depending on model.
                )
                #2. embed caption batch and store in db device.
                capembs = retrieval.embed_strings(
                        self.embedder,
                        cap,
                        layer_index = projconfig.layer_index,
                        embedding_size = projconfig.embedding_size,
                        device = self.embedder.device
                )
                #3. store it.
                self.docs.extend(cap)
                self.docembs.extend(capembs)
                torch.cuda.empty_cache()
        self.docembs = torch.stack(self.docembs).to(self.db_device)
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

    def generate_iterative(self, query):
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
            #2. for each iter of batch size. 
            interim_reps = []
            for i in range(0, len(top_k_docs), self.bs):
                #1. make a docstring of batch.
                docstring = "\n".join(top_k_docs[i:i + self.bs])
                #2. append topk docs to query to answer the question.
                augmented_query =  VizAG.kernel_prompt.format(docs = docstring, query = query)
                #3. generate message using generator.
                genmsg = generation.get_reply(self.gen_model, self.gen_tok, augmented_query)
                interim_reps.append(genmsg)
            #3. combine the inter replies to 1 string.
            combined_interim_strings = ""
            for i, reply in enumerate(interim_reps):
                combined_interim_strings += f"reply {i}: '{reply}'.\n"
            #4. append topk docs to query to answer the question.
            combined_augmented_query =  VizAG.kernel_combine_prompt.format(docs = combined_interim_strings, query = query)
            #5. return final message.
            finalmsg = generation.get_reply(self.gen_model, self.gen_tok, combined_augmented_query)
        return finalmsg



if __name__ == "__main__":
    #0. Get Args.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default = 'base', type = str)

    args = parser.parse_args()
    #1. Extra imports.
    from transformers import AutoProcessor, AutoModelForCausalLM
    from angle_emb import AnglE
    #2. get clip stuff.
    if args.model == 'base':
        clipproc = AutoProcessor.from_pretrained(projconfig.clip_model_name, cache_dir = projconfig.modelstore)
        clipmodel = AutoModelForCausalLM.from_pretrained(projconfig.clip_model_name, cache_dir = projconfig.modelstore).to(projconfig.device)
    elif args.model == 'sforce':
        clipmodel, clipproc = sforceclip.get_sforce_clip()
    else:
        exit("Bye!")
    print("CLIP loaded.")
    #3. get retriever.
    retriever = AnglE.from_pretrained(
        projconfig.emb_model_name,
        pooling_strategy=projconfig.emb_pooling_strategy,
        cache_dir = projconfig.modelstore
    ).to(projconfig.device)
    print("Retriever loaded.")
    #4. get generator.
    gen_tok = generation.get_tokenizer(projconfig.llm_name)
    gen_model = generation.get_model(projconfig.llm_name, projconfig.qtype, projconfig.device)
    print("Generator loaded.")
    #5. images.

    # images = [
    #     "https://static.toiimg.com/photo/msid-61220500,width-96,height-65.cms",
    #     "https://lp-cms-production.imgix.net/2021-01/GettyRF_450207051.jpg",
    #     "https://cdn.britannica.com/74/114874-050-6E04C88C/North-Face-Mount-Everest-Tibet-Autonomous-Region.jpg",
    #     "https://i.natgeofe.com/n/28fd84d5-ebb6-45c8-a758-17d0089a0464/TT_report_ST0058057copy-Enhanced_HR_4x3.jpg",
    #     "https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Swiss_Alps.jpg/1200px-Swiss_Alps.jpg"
    # ]
    transform = transforms.Resize((256,256))
    
    # Apply the transformation
    image_files = [f for f in os.listdir('/nfs_share2/ujjawal/VizAG/Flickr8k_Dataset/Flicker8k_Dataset')]
    images = []
    for filename in image_files:
        img_path = os.path.join('/nfs_share2/ujjawal/VizAG/Flickr8k_Dataset/Flicker8k_Dataset', filename)
        img = Image.fromarray(imageio.imread(img_path))
        resized_image = transform(img)
        images.append(resized_image)

    #6. make CLIP Vizag.
    vizag = VizAG(
        images, #images.
        clipproc, clipmodel, #clip stuff.
        retriever, #retriever.
        gen_tok, gen_model, #generator.
        k = len(image_files),
        img2cap = baseclip.img2cap
    )
    if os.path.exists(projconfig.flikr8k):
        vizag.setup_snapshot(projconfig.flikr8k)
    else:
        vizag.setup() #a fresh setup.
        vizag.save_snapshot(projconfig.flikr8k) #save snapshot. Setup takes 6+ minutes man! :'(

    #7. Iterative + combine to generate final answer.
    reply = vizag.generate_iterative("How many tables are present in the data?")
    print(f"{reply}")
    
