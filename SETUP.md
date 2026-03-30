# NutriVision — GitHub Setup Guide

## Step 1: Create the GitHub Repo

1. Go to https://github.com/new
2. Name it `NutriVision`
3. Set to **Public** or **Private** (your choice)
4. Do NOT initialize with README, .gitignore, or license (we already have them)
5. Click **Create repository**

## Step 2: Initialize and Push

After downloading and extracting the project folder, run these commands:

```bash
cd NutriVision

# Initialize git
git init
git branch -M main

# Stage all files
git add .

# First commit
git commit -m "Initial commit: NutriVision project structure and Level 1 pipeline"

# Connect to GitHub (replace with your actual repo URL)
git remote add origin https://github.com/YOUR_USERNAME/NutriVision.git

# Push
git push -u origin main
```

## Recommended Commit Strategy

As you work through the project, commit at these milestones:

```bash
# After Level 1 training completes
git add results/ notebooks/
git commit -m "Level 1: ResNet-50 and MobileNetV3 training results"

# After Level 2 nutrition mapping
git add src/nutrition.py notebooks/Level2_Nutrition_Pipeline.ipynb
git commit -m "Level 2: USDA nutrition mapping and end-to-end pipeline"

# After Level 3 demo
git add notebooks/Level3_Demo.ipynb
git commit -m "Level 3: Gradio demo with Grad-CAM and meal logging"

# Reports
git add docs/
git commit -m "Add Project 2 report"
```

## Keeping Model Weights Available

Model `.pth` files are git-ignored (too large for GitHub). Options:
- **Google Drive**: Save models to Drive from Colab and share the link in README
- **GitHub Releases**: Upload `.pth` files as release assets (up to 2GB each)
- **Hugging Face Hub**: Free model hosting with version control
