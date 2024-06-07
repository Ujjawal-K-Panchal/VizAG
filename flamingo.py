"""
Title: Test OpenFlamingo: `https://github.com/mlfoundations/open_flamingo`.

<!>: DOES NOT WORK!

Date: May 27, 2024; 7:22 PM

Author: Ujjawal K. Panchal & Ajinkya Chaudhari & Isha S. Joglekar
"""
from typing import List
import imageio

import torch
from huggingface_hub import hf_hub_download

from open_flamingo import create_model_and_transforms
import projconfig, baseclip


def load_flamingo():
    #1. download if not exists. 
    cpath = hf_hub_download(
        projconfig.flamingo_model_name,
        projconfig.flamingo_cpoint,
        cache_dir = projconfig.modelstore,
        token = projconfig.hf_token
    )
    #2. load and return model.
    model, processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-7b",
        tokenizer_path="anas-awadalla/mpt-7b",
        cross_attn_every_n_layers=4,
        cache_dir=projconfig.modelstore  # Defaults to ~/.cache
    )
    model.load_state_dict(torch.load(cpath), strict = True)
    return model, processor, tokenizer

def img2cap(processor, model, tokenizer, img, device, max_seq_len: int = projconfig.max_seq_len, KCAPTIONCMD = projconfig.KFLAMINGOCAPTIONCMD):
    """
    Desc:
        Image to Caption for Flamingo Model.
    Args:
        processor, model, tokenizer, img, device, max_seq_len.
    """
    #1. preprocess images.
    img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    img = [img] if not isinstance(img, List) else img
    #2. process images.
    inputs = [processor(images = i, return_tensors = "pt").unsqueeze(0) for i in img]
    inputs = torch.cat(inputs, dim = 0).unsqueeze(1).unsqueeze(0).to(device)
    #3. tokenize command.
    tokenizer.padding_side = "left"
    tok_cmd = tokenizer(
        [KCAPTIONCMD],
        return_tensors="pt",
    )
    #4. generate text.
    genids = model.generate(
        vision_x=inputs,
        lang_x=tok_cmd["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=max_seq_len,
    )
    gencap = tokenizer.batch_decode(genids, skip_special_tokens = True)
    return gencap
    

if __name__ == "__main__":
    #1. dl flamingo if not already downloaded.
    flamingo, processor, tok = load_flamingo()
    #2. caption some images.
    sample_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Ravivarmapress.jpg/800px-Ravivarmapress.jpg"
    image = [imageio.imread(sample_url)] * 2
    gencap = img2cap(processor, model, tokenizer, image, projconfig.device, 100)
    print(gencap)
