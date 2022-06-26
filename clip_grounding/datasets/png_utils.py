"""Helper functions for Panoptic Narrative Grounding."""

import os
from os.path import join, isdir, exists

import torch
from PIL import Image
from skimage import io
import numpy as np
import textwrap
import matplotlib.pyplot as plt
from matplotlib import transforms
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def rainbow_text(x,y,ls,lc,fig, ax,**kw):
    """
    Take a list of strings ``ls`` and colors ``lc`` and place them next to each
    other, with text ls[i] being shown in color lc[i].
    
    Ref: https://stackoverflow.com/questions/9169052/partial-coloring-of-text-in-matplotlib
    """
    t = ax.transAxes

    for s,c in zip(ls,lc):
        
        text = ax.text(x,y,s+" ",color=c, transform=t, **kw)
        text.draw(fig.canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(text._transform, x=ex.width, units='dots')


def find_first_index_greater_than(elements, key):
    return next(x[0] for x in enumerate(elements) if x[1] > key)


def split_caption_phrases(caption_phrases, colors, max_char_in_a_line=50):
    char_lengths = np.cumsum([len(x) for x in caption_phrases])
    thresholds = [max_char_in_a_line * i for i in range(1, 1 + char_lengths[-1] // max_char_in_a_line)]

    utt_per_line = []
    col_per_line = []
    start_index = 0
    for t in thresholds:
        index = find_first_index_greater_than(char_lengths, t)
        utt_per_line.append(caption_phrases[start_index:index])
        col_per_line.append(colors[start_index:index])
        start_index = index

    return utt_per_line, col_per_line


def show_image_and_caption(image: Image, caption_phrases: list, colors: list = None):

    if colors is None:
        colors = ["black" for _ in range(len(caption_phrases))]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    ax = axes[0]
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    utt_per_line, col_per_line = split_caption_phrases(caption_phrases, colors, max_char_in_a_line=50)
    y = 0.7
    for U, C in zip(utt_per_line, col_per_line):
        rainbow_text(
            0., y,
            U,
            C,
            size=15, ax=ax, fig=fig,
            horizontalalignment='left',
            verticalalignment='center',
        )
        y -= 0.11

    ax.axis("off")

    fig.tight_layout()
    plt.show()


