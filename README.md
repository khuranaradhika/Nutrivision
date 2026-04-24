# NutriVision

**Image-Based Food Recognition and Nutritional Estimation**

EECE 5639: Computer Vision | Spring 2026 | Northeastern University  
Radhika Khurana

---

## Overview

NutriVision is a deep learning pipeline that takes a photo of food and identifies what it is, then estimates nutritional content (calories, protein, carbs, fat) by mapping predictions to the USDA FoodData Central database. The goal is a lightweight, end-to-end system that could eventually run on-device for meal logging.

## Project Levels

| Level | Description | Status |
|-------|-------------|--------|
| Level 1 | Food classification on Food-101 (ResNet-50 and MobileNetV3) | Done |
| Level 2 | Nutrition mapping + multi-food handling + improved training | Done |
| Level 3 | Grad-CAM explainability + Gradio interactive demo | Done |

## How to Run

**Google Colab (recommended):** Open any notebook in the `notebooks/` directory, set runtime to GPU (T4), and run all cells.

**Local:**
```bash
pip install -r requirements.txt
jupyter notebook notebooks/Level1_Classification.ipynb     # Level 1
jupyter notebook notebooks/Level2_Nutrition_Pipeline.ipynb # Level 2
jupyter notebook notebooks/Level3_Demo.ipynb             # Level 3 + Gradio demo
```

Run Level 1 first — it saves `resnet50_food101_best.pth` and `mobilenetv3_food101_best.pth` to `models/`, which Levels 2 and 3 load.

## Results (Level 1)

| Model | Top-1 Acc | Top-5 Acc | Params | Inference |
|-------|-----------|-----------|--------|-----------|
| ResNet-50 | 82.95% | 96.18% | 23.7M | 5.6 ms |
| MobileNetV3-Large | 76.25% | 93.43% | 4.3M | 4.6 ms |

Both models were trained with two-phase transfer learning: the backbone was frozen for 5 epochs while the classification head was trained, then the full network was fine-tuned for up to 15 more epochs with differential learning rates (backbone 1e-5, head 1e-4). ResNet-50 converged to ~83% Top-1 accuracy on the Food-101 test set. MobileNetV3 is about 5.5x smaller and ~20% faster, though accuracy drops by ~7 points.

The hardest classes to distinguish were visually similar pairs — steak vs. filet mignon (50 errors), tuna tartare vs. beef tartare (33), and chocolate cake vs. chocolate mousse (23).

## Project Structure

```
NutriVision/
├── notebooks/
│ ├── Level1_Classification.ipynb      # training + evaluation
│ ├── Level2_Nutrition_Pipeline.ipynb  # nutrition pipeline + advanced training
│ └── Level3_Demo.ipynb               # Grad-CAM + Gradio demo
├── src/
│ ├── dataset.py     # Food-101 loading and transforms
│ ├── models.py      # ResNet-50 and MobileNetV3 setup
│ ├── train.py       # two-phase training loop
│ ├── evaluate.py    # metrics and prediction utilities
│ ├── utils.py       # plotting helpers
│ └── nutrition.py   # USDA nutrition database + lookup functions
├── configs/
│ └── default.yaml
├── results/
│ ├── figures/  # training curves, confusion matrices, Grad-CAM
│ └── metrics/  # classification reports, results JSON
├── models/     # saved .pth weights (git-ignored)
├── docs/       # proposal and final research paper
│   ├── Nutrivision_Final_Paper.pdf    # IEEE conference format final paper
│   ├── proposal.pdf                    # initial project proposal
│   └── Nutrivision_Final_Paper.tex    # LaTeX source
├── data/       # Food-101 dataset (downloaded automatically)
├── requirements.txt
└── README.md
```

## Dataset

[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) — 101 food categories, 101,000 images (750 train / 250 test per class). Automatically downloaded via `torchvision` when running the notebooks.

## Documentation

- **Final Paper**: [docs/Nutrivision_Final_Paper.pdf](docs/Nutrivision_Final_Paper.pdf) - Complete IEEE conference format paper documenting methodology, results, and analysis
- **Project Proposal**: [docs/proposal.pdf](docs/proposal.pdf) - Initial project proposal and timeline

## References

1. Bossard et al., "Food-101 — Mining Discriminative Components with Random Forests," ECCV 2014
2. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016
3. Howard et al., "Searching for MobileNetV3," ICCV 2019