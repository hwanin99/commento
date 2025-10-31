#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import glob
import pandas as pd


def make_metadata(original_img_path: str, edited_img_path: str, desc_path: str, save_path: str):
    original_imgs = sorted(glob.glob(original_img_path),key=lambda x: int(x.split('/')[-1].split('.')[0]))
    edited_imgs = sorted(glob.glob(edited_img_path),key=lambda x: int(x.split('/')[-1].split('.')[0]))

    df = pd.read_csv(desc_path) #Change to your modified prompt csv file path
    edit_prompts = df['modified_prompt'].to_list()

    dataset = []

    for original, edited, edit_prompt in zip(original_imgs, edited_imgs,edit_prompts):
        data_entry = {
            "original_image": original,
            "edited_image": edited,
            "edit_prompt": edit_prompt
        }
        dataset.append(data_entry)

    os.makedirs(save_path, exist_ok=True)

    # dataset
    with open(f'{save_path}/metadata.jsonl', 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Dataset creation complete! Total pairs: {len(dataset)}")

