#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display
from PIL import Image

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image


def generate(checkpoint_path, image_path, prompt, gradio=False):
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            checkpoint_path, torch_dtype=torch.float16,
            safety_checker = None,
            requires_safety_checker = False
        ).to("cuda:0")

    image = load_image(image_path)
    image = pipeline(prompt,image=image).images[0]

    if gradio:
        return image
        
    display(image)

