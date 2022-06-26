"""Image operations."""
from PIL import Image


def center_crop(im: Image):
    width, height = im.size
    new_width = width if width < height else height
    new_height = height if height < width else width 

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    
    return im


def pad_to_square(im: Image, color=(0, 0, 0)):
    width, height = im.size

    vert_pad = (max(width, height) - height) // 2
    hor_pad = (max(width, height) - width) // 2
    
    return add_margin(im, vert_pad, hor_pad, vert_pad, hor_pad, color=color)


def add_margin(pil_img, top, right, bottom, left, color=(0, 0, 0)):
    """Ref: https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/"""
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result
