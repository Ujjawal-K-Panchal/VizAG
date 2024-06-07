"""
Title: Test OpenFlamingo: `https://github.com/mlfoundations/open_flamingo`.

Date: May 27, 2024; 7:22 PM

Author: Ujjawal K. Panchal & Ajinkya Chaudhari & Isha S. Joglekar
"""
import imageio

from transformers import BlipProcessor, BlipForConditionalGeneration

import baseclip, projconfig

def get_sforce_clip(device = projconfig.device):
    #1. get model and processor.
    model = BlipForConditionalGeneration.from_pretrained(
        projconfig.sforceclip_model_name,
        cache_dir = projconfig.modelstore,
        token = projconfig.hf_token
    ).to(device)
    #2. get processor.
    processor = BlipProcessor.from_pretrained(
        projconfig.sforceclip_model_name,
        cache_dir = projconfig.modelstore,
        token = projconfig.hf_token
    )
    return model, processor

if __name__ == "__main__":
    #1. get model and processor.
    model, processor = get_sforce_clip()
    #2. caption the image.
    sample_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Ravivarmapress.jpg/800px-Ravivarmapress.jpg"
    image = [imageio.imread(sample_url)] * 2
    gencap = baseclip.img2cap(processor, model, image, projconfig.device, 100)
    print(gencap)