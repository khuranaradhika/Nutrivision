# NutriVision 🍽️

**Image-Based Food Recognition and Nutritional Estimation Using Deep Learning**

EECE 5639: Computer Vision | Spring 2026 | Northeastern University

Radhika Khurana

---

## Overview

NutriVision is a computer vision pipeline that takes a photograph of a meal, classifies the food item(s) present, and estimates nutritional content (calories, protein, carbohydrates, fat) by mapping predictions to the USDA FoodData Central database.

## Project Levels

| Level | Description | Status |
|-------|-------------|--------|
| **Level 1** | Food image classification with ResNet-50 and MobileNetV3 on Food-101 | 🔄 In Progress |
| **Level 2** | Nutrition mapping pipeline + multi-food handling + advanced training | ⬜ Planned |
| **Level 3** | Interactive web demo + Grad-CAM + text refinement + meal logging | ⬜ Planned |

## Project Structure

```
NutriVision/
├── notebooks/                # Colab/Jupyter notebooks
│   ├── Level1_Classification.ipynb
│   ├── Level2_Nutrition_Pipeline.ipynb
│   └── Level3_Demo.ipynb
├── src/                      # Reusable source code
│   ├── __init__.py
│   ├── dataset.py            # Data loading and transforms
│   ├── models.py             # Model definitions
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Evaluation and metrics
│   ├── nutrition.py          # USDA nutrition mapping
│   └── utils.py              # Visualization and helpers
├── configs/                  # Training configurations
│   └── default.yaml
├── results/                  # Output artifacts (git-ignored except .gitkeep)
│   ├── figures/              # Plots, confusion matrices
│   └── metrics/              # JSON/CSV metric files
├── models/                   # Saved model weights (git-ignored)
├── data/                     # Dataset directory (git-ignored)
├── docs/                     # Reports and proposal
│   └── proposal.pdf
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE
```

## Dataset

[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) — 101 food categories, 101,000 images (750 train / 250 test per class).

The dataset is **not** included in this repository. It will be downloaded automatically when running the notebooks.

## Setup

### Google Colab (Recommended)
1. Open `notebooks/Level1_Classification.ipynb` in Google Colab
2. Set runtime to **GPU** (`Runtime` → `Change runtime type` → `T4 GPU`)
3. Run all cells

### Local Setup
```bash
git clone https://github.com/YOUR_USERNAME/NutriVision.git
cd NutriVision
pip install -r requirements.txt
```

## Models

| Model | Top-1 Acc | Top-5 Acc | Params | Inference |
|-------|-----------|-----------|--------|-----------|
| ResNet-50 | TBD | TBD | 25.6M | TBD ms |
| MobileNetV3-Large | TBD | TBD | 5.4M | TBD ms |

## Key Technologies

- **Framework:** PyTorch
- **Models:** ResNet-50, MobileNetV3 (ImageNet pre-trained)
- **Dataset:** Food-101
- **Nutrition Data:** USDA FoodData Central
- **Explainability:** Grad-CAM (Level 3)
- **Demo:** Gradio / Streamlit (Level 3)

## Timeline

| Date | Milestone |
|------|-----------|
| Mar 24–Apr 9 | Level 1 — Classification pipeline |
| Apr 10–Apr 24 | Level 2 — Nutrition estimation |
| Apr 18–Apr 24 | Level 3 — Interactive demo (extra credit) |

## References

1. Bossard et al., "Food-101 — Mining Discriminative Components with Random Forests," ECCV 2014
2. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016
3. Howard et al., "Searching for MobileNetV3," ICCV 2019
4. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks," ICCV 2017

## License

This project is for academic use as part of EECE 5639 at Northeastern University.
