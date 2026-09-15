# DR Grading XAI

This repository contains a diabetic retinopathy grading project with explainable AI (XAI) experiments using several deep learning backbones. The goal is to classify retinal fundus images into severity levels and analyze model decisions using explainability methods.

## Project overview

The project compares multiple pretrained architectures for diabetic retinopathy detection, including:

- ResNet-50
- DenseNet-121
- EfficientNet-B3
- ViT Base-16
- DeiT Base-16

The workflow includes:

1. Data preprocessing
2. Model training and benchmarking
3. Evaluation using classification metrics
4. Visualization of results and model explanations

## Repository structure

```text
DR Grading XAI/
├── app/                             
├── checkpoints/                     # Saved model weights
│   ├── deit_base16.pth
│   ├── densenet121.pth
│   ├── efficientnet_b3.pth
│   ├── resnet50.pth
│   └── vit_base16.pth
├── data_preprocessing/
│   └── preprocessing-aptos-2019.ipynb
├── models_notebooks/
│   ├── deit-base-16.ipynb
│   ├── densenet-121.ipynb
│   ├── efficientnet-b3.ipynb
│   ├── resnet-50.ipynb
│   └── vit-base-16.ipynb
├── results/
│   ├── figures/
│   │   ├── deit_base16/
│   │   ├── densenet121/
│   │   ├── efficientnet_b3/
│   │   ├── resnet50/
│   │   └── vit_base16/
│   └── metrics/
│       ├── deit_base16_metrics.json
│       ├── densenet121_metrics.json
│       ├── efficientnet_b3_metrics.json
│       ├── Resnet50_metrics.json
│       └── vit_base16_metrics.json
├── README.md
└── .gitignore (if present)
```

## Data

The preprocessing notebook is designed for the APTOS 2019 retinal image dataset. It prepares the data for training and validation before model experimentation.

## Environment setup

It is recommended to use Python 3.10 or newer.

### Install dependencies

```bash
pip install torch torchvision torchaudio
pip install matplotlib seaborn scikit-learn pandas numpy
pip install opencv-python pillow jupyter
pip install timm
```

Depending on your setup, you may also need:

```bash
pip install grad-cam captum albumentations
```

## Typical workflow

1. Open the preprocessing notebook in `data_preprocessing/`.
2. Run the notebook to prepare the dataset.
3. Open the model notebook corresponding to the architecture you want to evaluate.
4. Train or load the model checkpoint from `checkpoints/`.
5. Inspect evaluation outputs under `results/metrics/`.
6. Review generated visual explanations under `results/figures/`.

## Model checkpoints

Pretrained or fine-tuned model weights are stored in `checkpoints/` and can be reused for evaluation or further experiments.

## Results

Model metrics are saved as JSON under `results/metrics/` and visual outputs are stored under `results/figures/`.

These outputs can be used to compare:

- Accuracy
- Loss
- Precision
- Recall
- F1-score
- Confusion matrix summaries
- Grad-CAM or XAI visualizations

## Notes

- This project is intended for research and experimentation.
- The code is notebook-driven and best suited for exploratory deep learning workflows.
- For production use, further validation, data auditing, and deployment checks are recommended.

## License

This project does not currently include a specified license file. If you plan to share or publish this repository, add an appropriate license before distribution.

