# Crossmodal Bias Mitigation

Research notebooks validating two frozen face-analysis models, **FairFace**
(demography) and **EmoNet** (emotion), for demographic bias, and training a small
PyTorch "correction" model on top of their logits to reduce gender bias in facial
analysis.

The work lives in Jupyter notebooks:

- `notebooks/251023_fairface_validation.ipynb` — compact pipeline: runs EmoNet +
  FairFace over cropped face images and writes the `*_emonet*.csv` artifacts, with
  precomputed logits.
- `notebooks/model_train.ipynb` — trains/evaluates the `GenderEmotionModel`
  correction model and produces the accuracy-gap figures.

## Data

We do not provide any of the FACES or ADFES image files. The precomputed demographic
logits and baseline demographic information is included in the `CSV` files in the 
`/data` folder for reproducibility.

## Environment

Managed with [uv](https://docs.astral.sh/uv/) (Python >= 3.13). Run `uv sync` and
`uv run jupyter lab`. The `external/fairface` and `external/emonet` models are git
submodules; run `git submodule update --init --recursive` after cloning. Datasets
and pretrained weights live outside the repo (see `AGENTS.md` for details).

## Publication

This work was published in the **IEEE Conference on Artificial Intelligence (CAI)**:

> Iris Dominguez-Catena; Daniel Paternain; Aranzazu Jurio; Mikel Galar.
> *Leveraging Cross-Modal Information to Reduce Gender Bias in Facial Analysis Systems.*

Link to the paper: <https://ieeexplore.ieee.org/document/11536269>
