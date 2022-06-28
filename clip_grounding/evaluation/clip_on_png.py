"""Evaluates cross-modal correspondence of CLIP on PNG images."""

import os
import sys
from os.path import join, exists

import warnings
warnings.filterwarnings('ignore')

from clip_grounding.utils.paths import REPO_PATH
sys.path.append(join(REPO_PATH, "CLIP_explainability/Transformer-MM-Explainability/"))

import torch
import CLIP.clip as clip
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from captum.attr import visualization
from torchmetrics import JaccardIndex
from collections import defaultdict
from IPython.core.display import display, HTML
from skimage import filters

from CLIP_explainability.utils import interpret, show_img_heatmap, show_txt_heatmap, color, _tokenizer
from clip_grounding.datasets.png import PNG
from clip_grounding.utils.image import pad_to_square
from clip_grounding.utils.visualize import show_grid_of_images
from clip_grounding.utils.log import tqdm_iterator, print_update



def show_cam(mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap
    cam = cam / np.max(cam)
    return cam


def interpret_and_generate(model, img, texts, orig_image, return_outputs=False, show=True):
    text = clip.tokenize(texts).to(device)
    R_text, R_image = interpret(model=model, image=img, texts=text, device=device)
    batch_size = text.shape[0]
    
    outputs = []
    for i in range(batch_size):
        text_scores, text_tokens_decoded = show_txt_heatmap(texts[i], text[i], R_text[i], show=show)
        image_relevance = show_img_heatmap(R_image[i], img, orig_image=orig_image, device=device, show=show)
        plt.show()
        outputs.append({"text_scores": text_scores, "image_relevance": image_relevance, "tokens_decoded": text_tokens_decoded})
    
    if return_outputs:
        return outputs


def process_entry_text_to_image(entry, unimodal=False):
    image = entry['image']
    text_mask = entry['text_mask']
    text = entry['text']
    orig_image = pad_to_square(image)
    
    img = preprocess(orig_image).unsqueeze(0).to(device)
    text_index = text_mask.argmax()
    texts = [text[text_index]] if not unimodal else ['']
    
    return img, texts, orig_image


def preprocess_ground_truth_mask(mask, resize_shape):
    mask = Image.fromarray(mask.astype(np.uint8) * 255)
    mask = pad_to_square(mask, color=0)
    mask = mask.resize(resize_shape)
    mask = np.asarray(mask) / 255.
    return mask


def apply_otsu_threshold(relevance_map):
    threshold = filters.threshold_otsu(relevance_map)
    otsu_map = (relevance_map > threshold).astype(np.uint8)
    return otsu_map


def evaluate_text_to_image(method, dataset, debug=False):

    instance_level_metrics = defaultdict(list)
    entry_level_metrics = defaultdict(list)
    
    jaccard = JaccardIndex(num_classes=2)
    jaccard = jaccard.to(device)

    num_iter = len(dataset)
    if debug:
        num_iter = 100

    iterator = tqdm_iterator(range(num_iter), desc=f"Evaluating on {type(dataset).__name__} dataset")
    for idx in iterator:
        instance = dataset[idx]
        
        instance_iou = 0.
        for entry in instance:
            
            # preprocess the image and text
            unimodal = True if method == "clip-unimodal" else False
            test_img, test_texts, orig_image = process_entry_text_to_image(entry, unimodal=unimodal)

            if method in ["clip", "clip-unimodal"]:
                
                # compute the relevance scores 
                outputs = interpret_and_generate(model, test_img, test_texts, orig_image, return_outputs=True, show=False)
                
                # use the image relevance score to compute IoU w.r.t. ground truth segmentation masks

                # NOTE: since we pass single entry (1-sized batch), outputs[0] contains our reqd outputs
                relevance_map = outputs[0]["image_relevance"]
            elif method == "random":
                relevance_map = np.random.uniform(low=0., high=1., size=tuple(test_img.shape[2:]))
                
            otsu_relevance_map = apply_otsu_threshold(relevance_map)
            
            ground_truth_mask = entry["image_mask"]
            ground_truth_mask = preprocess_ground_truth_mask(ground_truth_mask, relevance_map.shape)
            
            entry_iou = jaccard(
                torch.from_numpy(otsu_relevance_map).to(device),
                torch.from_numpy(ground_truth_mask.astype(np.uint8)).to(device),
            )
            entry_iou = entry_iou.item()
            instance_iou += (entry_iou / len(entry))
            
            entry_level_metrics["iou"].append(entry_iou)
        
        # capture instance (image-sentence pair) level IoU
        instance_level_metrics["iou"].append(instance_iou)
    
    average_metrics = {k: np.mean(v) for k, v in entry_level_metrics.items()}
    
    return (
        average_metrics,
        instance_level_metrics,
        entry_level_metrics
    )


def process_entry_image_to_text(entry, unimodal=False):
    
    if not unimodal:
        if len(np.asarray(entry["image"]).shape) == 3:
            mask = np.repeat(np.expand_dims(entry['image_mask'], -1), 3, axis=-1)
        else:
            mask = np.asarray(entry['image_mask'])

        masked_image = (mask * np.asarray(entry['image'])).astype(np.uint8)
        masked_image = Image.fromarray(masked_image)
        orig_image = pad_to_square(masked_image)
        img = preprocess(orig_image).unsqueeze(0).to(device)
    else:
        orig_image_shape = max(np.asarray(entry['image']).shape[:2])
        orig_image = Image.fromarray(np.zeros((orig_image_shape, orig_image_shape, 3), dtype=np.uint8))
        # orig_image = Image.fromarray(np.random.randint(0, 256, (orig_image_shape, orig_image_shape, 3), dtype=np.uint8))
        img = preprocess(orig_image).unsqueeze(0).to(device)
    
    texts = [' '.join(entry['text'])]

    return img, texts, orig_image


def process_text_mask(text, text_mask, tokens):

    token_level_mask = np.zeros(len(tokens))

    for label, subtext in zip(text_mask, text):

        subtext_tokens=_tokenizer.encode(subtext)
        subtext_tokens_decoded=[_tokenizer.decode([a]) for a in subtext_tokens]

        if label == 1:
            start = tokens.index(subtext_tokens_decoded[0])
            end = tokens.index(subtext_tokens_decoded[-1])
            token_level_mask[start:end + 1] = 1

    return token_level_mask


def evaluate_image_to_text(method, dataset, debug=False, clamp_sentence_len=70):

    instance_level_metrics = defaultdict(list)
    entry_level_metrics = defaultdict(list)
    
    # skipped if text length > 77 which is CLIP limit
    num_entries_skipped = 0
    num_total_entries = 0
    
    num_iter = len(dataset)
    if debug:
        num_iter = 100
    
    jaccard_image_to_text = JaccardIndex(num_classes=2).to(device)

    iterator = tqdm_iterator(range(num_iter), desc=f"Evaluating on {type(dataset).__name__} dataset")
    for idx in iterator:
        instance = dataset[idx]
        
        instance_iou = 0.
        for entry in instance:
            num_total_entries += 1
            
            # preprocess the image and text
            unimodal = True if method == "clip-unimodal" else False
            img, texts, orig_image = process_entry_image_to_text(entry, unimodal=unimodal)

            appx_total_sent_len = np.sum([len(x.split(" ")) for x in texts])
            if appx_total_sent_len > clamp_sentence_len:
                # print(f"Skipping an entry since it's text has appx"\
                # " {appx_total_sent_len} while CLIP cannot process beyond {clamp_sentence_len}")
                num_entries_skipped += 1
                continue
            
            # compute the relevance scores 
            if method in ["clip", "clip-unimodal"]:
                try:
                    outputs = interpret_and_generate(model, img, texts, orig_image, return_outputs=True, show=False)
                except:
                    num_entries_skipped += 1
                    continue
            elif method == "random":
                text = texts[0]
                text_tokens = _tokenizer.encode(text)
                text_tokens_decoded=[_tokenizer.decode([a]) for a in text_tokens]
                outputs = [
                    {
                        "text_scores": np.random.uniform(low=0., high=1., size=len(text_tokens_decoded)),
                        "tokens_decoded": text_tokens_decoded,
                    }
                ]
            
            # use the text relevance score to compute IoU w.r.t. ground truth text masks
            # NOTE: since we pass single entry (1-sized batch), outputs[0] contains our reqd outputs
            token_relevance_scores = outputs[0]["text_scores"]
            if isinstance(token_relevance_scores, torch.Tensor):
                token_relevance_scores = token_relevance_scores.cpu().numpy()
            token_relevance_scores = apply_otsu_threshold(token_relevance_scores)
            token_ground_truth_mask = process_text_mask(entry["text"], entry["text_mask"], outputs[0]["tokens_decoded"])
            
            entry_iou = jaccard_image_to_text(
                torch.from_numpy(token_relevance_scores).to(device),
                torch.from_numpy(token_ground_truth_mask.astype(np.uint8)).to(device),
            )
            entry_iou = entry_iou.item()

            instance_iou += (entry_iou / len(entry))
            entry_level_metrics["iou"].append(entry_iou)
        
        # capture instance (image-sentence pair) level IoU
        instance_level_metrics["iou"].append(instance_iou)
    
    print(f"CAUTION: Skipped {(num_entries_skipped / num_total_entries) * 100} % since these had length > 77 (CLIP limit).")
    average_metrics = {k: np.mean(v) for k, v in entry_level_metrics.items()}
    
    return (
        average_metrics,
        instance_level_metrics,
        entry_level_metrics
    )


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser("Evaluate Image-to-Text & Text-to-Image model")
    parser.add_argument(
        "--eval_method", type=str, default="clip",
        choices=["clip", "random", "clip-unimodal"],
        help="Evaluation method to use",
    )
    parser.add_argument(
        "--ignore_cache", action="store_true",
        help="Ignore cache and force re-generation of the results",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run evaluation on a small subset of the dataset",
    )
    args = parser.parse_args()
    
    print_update("Using evaluation method: {}".format(args.eval_method))
    
    
    clip.clip._MODELS = {
        "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    }
    
    # specify device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # load CLIP model
    print_update("Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    print()
    
    # load PNG dataset
    print_update("Loading PNG dataset...")
    dataset = PNG(dataset_root=join(REPO_PATH, "data", "panoptic_narrative_grounding"), split="val2017")
    print()
    
    # evaluate

    # save metrics
    metrics_dir = join(REPO_PATH, "outputs")
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_path = join(metrics_dir, f"{args.eval_method}_on_{type(dataset).__name__}_text2image_metrics.pt")
    if (not exists(metrics_path)) or args.ignore_cache:
        print_update("Computing metrics for text-to-image grounding")
        average_metrics, instance_level_metrics, entry_level_metrics = evaluate_text_to_image(
            args.eval_method, dataset, debug=args.debug,
        )
        metrics = {
            "average_metrics": average_metrics,
            "instance_level_metrics":instance_level_metrics,
            "entry_level_metrics": entry_level_metrics
        }

        torch.save(metrics, metrics_path)
        print("TEXT2IMAGE METRICS SAVED TO:", metrics_path)
    else:
        print(f"Metrics already exist at: {metrics_path}. Loading cached metrics.")
        metrics = torch.load(metrics_path)
        average_metrics = metrics["average_metrics"]
    print("TEXT2IMAGE METRICS:", np.round(average_metrics["iou"], 4))

    print()
    
    metrics_path = join(metrics_dir, f"{args.eval_method}_on_{type(dataset).__name__}_image2text_metrics.pt")
    if (not exists(metrics_path)) or args.ignore_cache:
        print_update("Computing metrics for image-to-text grounding")
        average_metrics, instance_level_metrics, entry_level_metrics = evaluate_image_to_text(
            args.eval_method, dataset, debug=args.debug,
        )
        
        torch.save(
            {
                "average_metrics": average_metrics,
                "instance_level_metrics":instance_level_metrics,
                "entry_level_metrics": entry_level_metrics
            },
            metrics_path,
        )
        print("IMAGE2TEXT METRICS SAVED TO:", metrics_path)
    else:
        print(f"Metrics already exist at: {metrics_path}. Loading cached metrics.")
        metrics = torch.load(metrics_path)
        average_metrics = metrics["average_metrics"]
    print("IMAGE2TEXT METRICS:", np.round(average_metrics["iou"], 4))
