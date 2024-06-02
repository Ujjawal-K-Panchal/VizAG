"""
Title: Test Clip.

Date: May 27, 2024; 7:22 PM

Author: Ujjawal K. Panchal & Ajinkya Chaudhari & Isha S. Joglekar
"""
from typing import Union, List
import requests, numpy as np, imageio.v2 as imageio
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCausalLM

import projconfig

def transforms(example_batch):
    images = [x for x in example_batch["image"]]
    captions = [x for x in example_batch["text"]]
    inputs = processor(images=images, text=captions, padding="max_length")
    inputs.update({"labels": inputs["input_ids"]})
    return inputs

def img2cap(
    processor, model,
    img: Union[Image, np.array, List[Image], List[np.array]],
    device: str = projconfig.device, max_tokens: int = 100
):
    """
    Desc:
        Image to Caption.
    Args:
        1. `img`: Image | Images.
        2. `device`: device.
        3. `max_tokens`: maximum number of tokens.
    """
    img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    inputs = processor(images = img, return_tensors = "pt").to(device)
    genids = model.generate(pixel_values = inputs.pixel_values, max_length = 100)
    gencap = processor.batch_decode(genids, skip_special_tokens = True)
    return gencap

if __name__ == "__main__":
    #1. load model.
    processor = AutoProcessor.from_pretrained(projconfig.clip_model_name, cache_dir = projconfig.modelstore)
    model = AutoModelForCausalLM.from_pretrained(projconfig.clip_model_name, cache_dir = projconfig.modelstore).to(projconfig.device)
    #2. caption the image.
    sample_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Ravivarmapress.jpg/800px-Ravivarmapress.jpg"
    image = [imageio.imread(sample_url)] * 2
    gencap = img2cap(processor, model, image, projconfig.device, 100)
    print(gencap)