# Setup

## Installation

1. Create and activate `conda` enviroment
    ```sh
    conda create -n clip-grounding -y python=3.9
    conda activate clip-grounding
    ```
2. Install dependencies
    ```sh
    pip install einops
    pip install ftfy
    pip install captum
    pip install torch torchvision
    pip install ipdb jupyterlab matplotlib numpy scipy scikit-learn tqdm natsort opencv-python pillow pyyaml scikit-image imgaug
    pip install ipywidgets widgetsnbextension pandas-profiling
    pip install ftfy regex captum
    pip install torchmetrics
    ```
3. Clone repo for [Transformer-MM-Explanability](https://github.com/hila-chefer/Transformer-MM-Explainability) inside the `CLIP_explainability/` folder.
    ```sh
    cd CLIP_explainability/
    git clone https://github.com/hila-chefer/Transformer-MM-Explainability.git
    ```