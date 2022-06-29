"""Converts notebook for qualitative results to a python script."""
import sys
from os.path import join

from clip_grounding.utils.paths import REPO_PATH
sys.path.append(join(REPO_PATH, "CLIP_explainability/Transformer-MM-Explainability/"))

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import CLIP.clip as clip
import cv2
from PIL import Image
from glob import glob
from natsort import natsorted

from clip_grounding.utils.paths import REPO_PATH
from clip_grounding.utils.io import load_json
from clip_grounding.utils.visualize import set_latex_fonts, show_grid_of_images
from clip_grounding.utils.image import pad_to_square
from clip_grounding.datasets.png_utils import show_images_and_caption
from clip_grounding.datasets.png import (
    PNG,
    visualize_item,
    overlay_segmask_on_image,
    overlay_relevance_map_on_image,
    get_text_colors,
)
from clip_grounding.evaluation.clip_on_png import (
    process_entry_image_to_text,
    process_entry_text_to_image,
    interpret_and_generate,
)

# load dataset
dataset = PNG(dataset_root=join(REPO_PATH, "data/panoptic_narrative_grounding"), split="val2017")

# load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)


def visualize_entry_text_to_image(entry, pad_images=True, figsize=(18, 5)):
    test_img, test_texts, orig_image = process_entry_text_to_image(entry, unimodal=False)
    outputs = interpret_and_generate(model, test_img, test_texts, orig_image, return_outputs=True, show=False)
    relevance_map = outputs[0]["image_relevance"]
    
    image_with_mask = overlay_segmask_on_image(entry["image"], entry["image_mask"])
    if pad_images:
        image_with_mask = pad_to_square(image_with_mask)
    
    image_with_relevance_map = overlay_relevance_map_on_image(entry["image"], relevance_map)
    if pad_images:
        image_with_relevance_map = pad_to_square(image_with_relevance_map)
    
    text_colors = get_text_colors(entry["text"], entry["text_mask"])
    
    show_images_and_caption(
        [image_with_mask, image_with_relevance_map],
        entry["text"], text_colors, figsize=figsize,
        image_xlabels=["Ground truth segmentation", "Predicted relevance map"]
    )


def create_and_save_gif(filenames, save_path, **kwargs):
    import imageio
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(save_path, images, **kwargs)
    

idx = 100
instance = dataset[idx]

instance_dir = join(REPO_PATH, "figures", f"instance-{idx}")
os.makedirs(instance_dir, exist_ok=True)

for i, entry in enumerate(instance):
    del entry["full_caption"]

    visualize_entry_text_to_image(entry, pad_images=False, figsize=(19, 4))
    
    save_path = instance_dir
    plt.savefig(join(instance_dir, f"viz-{i}.png"), bbox_inches="tight")


filenames = natsorted(glob(join(instance_dir, "viz-*.png")))
save_path = join(REPO_PATH, "media", "sample.gif")

create_and_save_gif(filenames, save_path, duration=3)
