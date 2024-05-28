"""
Title: Project Configuration.

Date: May 27, 2024; 7:22 PM

Author: Ujjawal K. Panchal & Ajinkya Chaudhari & Isha S. Joglekar
"""
import os

modelstore = "./modelstore"

#embedder config.
# it is recommended to set layers from 20 to 24.
layer_index = 22  # 1d: layer
embedding_size = 768  # 2d: embedding size

if not os.path.exists(modelstore):
    os.makedirs(modelstore)