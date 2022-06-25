"""Path helpers for the relfm project."""
from os.path import join, abspath, dirname


REPO_PATH = dirname(dirname(dirname(abspath(__file__))))
DATA_ROOT = join(REPO_PATH, "data")

DATASET_ROOTS = {
    "PNG": join(DATA_ROOT, "panoptic_narrative_grounding"),
}
