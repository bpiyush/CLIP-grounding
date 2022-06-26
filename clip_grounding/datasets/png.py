"""
Dataset object for Panoptic Narrative Grounding.

Paper: https://openaccess.thecvf.com/content/ICCV2021/papers/Gonzalez_Panoptic_Narrative_Grounding_ICCV_2021_paper.pdf
"""

import os
from os.path import join, isdir, exists

import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage import io
import numpy as np
import textwrap
import matplotlib.pyplot as plt
from matplotlib import transforms
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import matplotlib.colors as mc

from clip_grounding.utils.io import load_json
from clip_grounding.datasets.png_utils import show_image_and_caption


class PNG(Dataset):
    """Panoptic Narrative Grounding."""

    def __init__(self, dataset_root, split) -> None:
        """
        Initializer.
        
        Args:
            dataset_root (str): path to the folder containing PNG dataset
            split (str): MS-COCO split such as train2017/val2017
        """
        super().__init__()
        
        assert isdir(dataset_root)
        self.dataset_root = dataset_root
        
        assert split in ["val2017"], f"Split {split} not supported. "\
            "Currently, only supports split `val2017`."
        self.split = split
        
        self.ann_dir = join(self.dataset_root, "annotations")
        feat_dir = join(self.dataset_root, "features")
        
        panoptic = load_json(join(self.ann_dir, "panoptic_{:s}.json".format(split)))
        images = panoptic["images"]
        self.images_info = {i["id"]: i for i in images}
        panoptic_anns = panoptic["annotations"]
        self.panoptic_anns = {int(a["image_id"]): a for a in panoptic_anns}
        
        self.panoptic_pred_path = join(
            feat_dir, split, "panoptic_seg_predictions"
        )
        assert isdir(self.panoptic_pred_path)
                
        panoptic_narratives_path = join(self.dataset_root, "annotations", f"png_coco_{split}.json")
        self.panoptic_narratives = load_json(panoptic_narratives_path)
    
    def __len__(self):
        return len(self.panoptic_narratives)
    
    def get_image_path(self, image_id: str):
        image_path = join(self.dataset_root, "images", self.split, f"{image_id.zfill(12)}.jpg")
        return image_path

    def __getitem__(self, idx: int):
        narr = self.panoptic_narratives[idx]

        image_id = narr["image_id"]
        image_path = self.get_image_path(image_id)
        assert exists(image_path)

        image = Image.open(image_path)
        caption = narr["caption"]

        # show_single_image(image, title=caption, titlesize=12)

        segments = narr["segments"]

        image_id = int(narr["image_id"])
        panoptic_ann = self.panoptic_anns[image_id]
        panoptic_ann = self.panoptic_anns[image_id]
        segment_infos = {}
        for s in panoptic_ann["segments_info"]:
            idi = s["id"]
            segment_infos[idi] = s

            
        image_info = self.images_info[image_id]
        panoptic_segm = io.imread(
            join(
                self.ann_dir,
                "panoptic_segmentation",
                self.split,
                "{:012d}.png".format(image_id),
            )
        )
        panoptic_segm = (
            panoptic_segm[:, :, 0]
            + panoptic_segm[:, :, 1] * 256
            + panoptic_segm[:, :, 2] * 256 ** 2
        )

        panoptic_ann = self.panoptic_anns[image_id]
        panoptic_pred = io.imread(
            join(self.panoptic_pred_path, "{:012d}.png".format(image_id))
        )[:, :, 0]


        # # select a single utterance to visualize
        # segment = segments[7]
        # segment_ids = segment["segment_ids"]
        # segment_mask = np.zeros((image_info["height"], image_info["width"]))
        # for segment_id in segment_ids:
        #     segment_id = int(segment_id)
        #     segment_mask[panoptic_segm == segment_id] = 1.

        utterances = [s["utterance"] for s in segments]
        outputs = []
        for i, segment in enumerate(segments):
            
            # create segmentation mask on image
            segment_ids = segment["segment_ids"]
            
            # if no annotation for this word, skip
            if not len(segment_ids):
                continue
            
            segment_mask = np.zeros((image_info["height"], image_info["width"]))
            for segment_id in segment_ids:
                segment_id = int(segment_id)
                segment_mask[panoptic_segm == segment_id] = 1.
            
            # store the outputs
            text_mask = np.zeros(len(utterances))
            text_mask[i] = 1.
            segment_data = dict(
                image=image,
                text=utterances,
                image_mask=segment_mask,
                text_mask=text_mask,
                full_caption=caption,
            )
            outputs.append(segment_data)
            
            # # visualize segmentation mask with associated text
            # segment_color = "red"
            # segmap = SegmentationMapsOnImage(
            #     segment_mask.astype(np.uint8), shape=segment_mask.shape,
            # )
            # image_with_segmap = segmap.draw_on_image(np.asarray(image), colors=[0, COLORS[segment_color]])[0]
            # image_with_segmap = Image.fromarray(image_with_segmap)
            
            # colors = ["black" for _ in range(len(utterances))]
            # colors[i] = segment_color
            # show_image_and_caption(image_with_segmap, utterances, colors)

        return outputs


def visualize_item(image, text, image_mask, text_mask, segment_color="red"):
    
    segmap = SegmentationMapsOnImage(
        image_mask.astype(np.uint8), shape=image_mask.shape,
    )
    rgb_color = mc.to_rgb(segment_color)
    rgb_color = 255 * np.array(rgb_color)
    image_with_segmap = segmap.draw_on_image(np.asarray(image), colors=[0, rgb_color])[0]
    image_with_segmap = Image.fromarray(image_with_segmap)
    
    colors = ["black" for _ in range(len(text))]
    
    text_idx = text_mask.argmax()
    colors[text_idx] = segment_color
    show_image_and_caption(image_with_segmap, text, colors)



if __name__ == "__main__":
    from clip_grounding.utils.paths import REPO_PATH, DATASET_ROOTS
    
    PNG_ROOT = DATASET_ROOTS["PNG"]
    dataset = PNG(dataset_root=PNG_ROOT, split="val2017")

    item = dataset[0]
    sub_item = item[1]
    visualize_item(
        image=sub_item["image"],
        text=sub_item["text"],
        image_mask=sub_item["image_mask"],
        text_mask=sub_item["text_mask"],
        segment_color="red"
    )
