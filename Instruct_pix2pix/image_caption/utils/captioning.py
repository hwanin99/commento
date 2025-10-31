""" Setup """

import os, subprocess

def setup():
    install_cmds = [
        ['pip', 'install', 'open_clip_torch'],
        ['pip', 'install', 'clip-interrogator'],
    ]
    for cmd in install_cmds:
        print(subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8'))

setup()


caption_model_name = 'blip-large'  # ["blip-base", "blip-large", "git-large-coco"]
clip_model_name = 'ViT-L-14/openai'  # ["ViT-L-14/openai", "ViT-H-14/laion2b_s32b_b79k"]

from clip_interrogator import Config, Interrogator

config = Config()
config.clip_model_name = clip_model_name
config.caption_model_name = caption_model_name
ci = Interrogator(config)


def image_analysis(image):
    image = image.convert('RGB')
    image_features = ci.image_to_features(image) # 이미지를 특징 벡터로 변환

    # 이미지의 특징 벡터를 기반으로 다양한 항목들과의 유사도를 계산하여 상위 5개를 반환
    top_mediums = ci.mediums.rank(image_features, 5)
    top_artists = ci.artists.rank(image_features, 5)
    top_movements = ci.movements.rank(image_features, 5)
    top_trendings = ci.trendings.rank(image_features, 5)
    top_flavors = ci.flavors.rank(image_features, 5)

    # 각 항목에 대해 해당 항목과 이미지의 유사도를 계산하여 딕셔너리 형태로 반환
    medium_ranks = {medium: sim for medium, sim in zip(top_mediums, ci.similarities(image_features, top_mediums))}
    artist_ranks = {artist: sim for artist, sim in zip(top_artists, ci.similarities(image_features, top_artists))}
    movement_ranks = {movement: sim for movement, sim in zip(top_movements, ci.similarities(image_features, top_movements))}
    trending_ranks = {trending: sim for trending, sim in zip(top_trendings, ci.similarities(image_features, top_trendings))}
    flavor_ranks = {flavor: sim for flavor, sim in zip(top_flavors, ci.similarities(image_features, top_flavors))}

    return medium_ranks, artist_ranks, movement_ranks, trending_ranks, flavor_ranks


def image_to_prompt(image, mode):
    ci.config.chunk_size = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    ci.config.flavor_intermediate_count = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image)
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    elif mode == 'fast':
        return ci.interrogate_fast(image)
    elif mode == 'negative':
        return ci.interrogate_negative(image)



""" Batch process a directory of images """

import os
import csv
from PIL import Image
from tqdm import tqdm
from IPython.display import clear_output, display


def sanitize_for_filename(prompt: str, max_len: int) -> str:
    name = "".join(c for c in prompt if (c.isalnum() or c in ",._-! "))
    name = name.strip()[:(max_len-4)] # extra space for extension
    return name


def captioning(dir_path: str, prompt_mode='best', output_mode='desc.csv', max_filename_len=128):
    ci.config.quiet = True
    
    files = sorted([f for f in os.listdir(dir_path) if f.endswith('.jpg') or f.endswith('.png')],key=lambda x: int(x.split('/')[-1].split('.')[0])) if os.path.exists(dir_path) else []
    prompts = []
    for idx, file in enumerate(tqdm(files, desc='Generating prompts')):
        if idx > 0 and idx % 100 == 0:
            clear_output(wait=True)
    
        image = Image.open(os.path.join(dir_path, file)).convert('RGB')
        prompt = image_to_prompt(image, prompt_mode)
        prompts.append(prompt)
    
        print(prompt)
        thumb = image.copy()
        thumb.thumbnail([256, 256])
        display(thumb)
    
        if output_mode == 'rename':
            name = sanitize_for_filename(prompt, max_filename_len)
            ext = os.path.splitext(file)[1]
            filename = name + ext
            idx = 1
            while os.path.exists(os.path.join(dir_path, filename)):
                print(f'File {filename} already exists, trying {idx+1}...')
                filename = f"{name}_{idx}{ext}"
                idx += 1
            os.rename(os.path.join(dir_path, file), os.path.join(dir_path, filename))
    
    if len(prompts):
        if output_mode == 'desc.csv':
            csv_path = ('./dataset/desc.csv')
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                w.writerow(['image', 'prompt'])
                for file, prompt in zip(files, prompts):
                    w.writerow([file, prompt])
    
            print(f"\n\n\n\nGenerated {len(prompts)} prompts and saved to {csv_path}")
        else:
            print(f"\n\n\n\nGenerated {len(prompts)} prompts and renamed your files.")
    else:
        print(f"Sorry, I couldn't find any images in {dir_path}")
