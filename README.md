# CLIP-grounding
Quantitative evaluation of CLIP's interpretability in terms of cross-modal grounding.


## Setup

Follow the steps provided [here](./setup/README.md) to create a `conda` enviroment and activate it.

## Dataset

1. Download the MSCOCO dataset (only validation images are required for this work) and its panoptic segmentation annotations by running:
    ```console
    bash setup/download_mscoco.sh
    ```

    This shall result in the following folder structure:
    ```zsh
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
2. (Optional) Pre-process the data as instructed in the [official panoptic narrative grounding repo](https://github.com/BCV-Uniandes/PNG). See [Appendix A](#appA) for sample instructions to run this on a Mac CPU machine. This shall create `data/panoptic_narrative_grounding/annotations/png_coco_val2017_dataloader.json` file. Note that we run this step only for `val2017` split and not the training set.


## Demo

:date:  Coming soon!


## Quantitative evaluation

In order to reproduce our results of CLIP model on Panoptic Narrative Grounding (PNG) benchmark dataset, we use the following procedure:
* Activate `conda` enviroment and set `PYTHONPATH`. Make sure you are at the repo root.
    ```sh
    conda activate clip-grounding
    export PYTHONPATH=$PWD
    ```
* Run the evaluation script:
    ```sh
    python clip_grounding/evaluation/clip_on_png.py
    ```
    This shall save metrics in `outputs/` folder as well as print the results in the following form.
    ```console
    TEXT2IMAGE METRICS: {'iou': 0.4892}
    IMAGE2TEXT METRICS: {'iou': 0.5015}
    ```


## Appendix

### A. Pre-processing data based on Panoptic Narrative Grounding <a class="anchor" id="appA"></a>

* Clone the repo [PNG](https://github.com/BCV-Uniandes/PNG)
* Create `conda` environment and install dependencies:
    ```sh
    conda create -n png -y python=3.6
    conda activate png

    pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
    pip install tqdm scipy

    # This failed initially on Mac and needed
    # `brew install mpich` to be run before this
    pip install mpi4py
    pip install scikit-image
    pip install boto3 requests
    ```
* Change the script to only run for `val2017` split: Change line 34 in `data/pre_process.py` as:
    ```python
    # splits = ["train2017", "val2017"]
    splits = ["val2017"]
    ```
* Run the script:
    ```sh
    cd data/
    python pre_process.py --data_dir /path/to/CLIP-grounding/repo/data/panoptic_narrative_grounding/
    ```
    This step takes about 10 minutes to run (only for `val2017` split).


## Acknowledgements
