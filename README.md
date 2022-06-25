# CLIP-grounding
Evaluating interpretability of CLIP in terms of grounding.


## Dataset

1. Download the MSCOCO dataset and its panoptic segmentation annotations by running:
    ```sh
    bash setup/download_mscoco.sh
    ```

    This shall result in the following folder structure:
    ```sh
    data/panoptic_narrative_grounding
    ├── __MACOSX
    │   └── panoptic_val2017
    ├── annotations
    │   ├── panoptic_segmentation
    │   ├── panoptic_train2017.json
    │   ├── panoptic_val2017.json
    │   └── png_coco_val2017.json
    └── images
        └── val2017

    6 directories, 3 files
    ```

