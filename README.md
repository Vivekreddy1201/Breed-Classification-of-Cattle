# Indian Bovine Breed Classifier

A deep learning image classification model to identify Indian cattle and buffalo breeds using **EfficientNetB0** and transfer learning.

---

## Overview

This project combines two Kaggle datasets to build a multi-class classifier capable of recognizing **40+ breeds** of Indian bovines (cattle and buffalo). The model is trained on Google Colab with GPU acceleration.

---

## Datasets

| Dataset | Source | Size |

| Indian Bovine Breeds | `lukex9442/indian-bovine-breeds` (Kaggle) | ~2.84 GB |
| Indian Buffalo Dataset | `atharvadarpude/indian-buffalo-dataset` (Kaggle) | ~38.4 MB |

Both datasets are downloaded via `kagglehub`, normalized, and merged into a single `final_dataset/` directory.

---

## Breeds Covered (Sample)

`alambadi`, `amritmahal`, `ayrshire`, `banni`, `bargur`, `bhadawari`, `brown_swiss`, `chhattisgarhi`, `chilika`, `dangi`, `deoni`, `gir`, `gojri`, `guernsey`, `hallikar`, `hariana`, `holstein_friesian`, `jaffarabadi`, `jersey`, `kangayam`, `kankrej`, `khillari`, `krishna_valley`, and more.

---

## Model Architecture

- **Base model:** EfficientNetB0 (pretrained on ImageNet)
- **Custom head:** GlobalAveragePooling2D → Dense → Dropout → Softmax output
- **Optimizer:** Adam
- **Input preprocessing:** EfficientNet-specific `preprocess_input`

### Callbacks
- `ModelCheckpoint` — saves the best model
- `EarlyStopping` — stops training when validation loss plateaus
- `ReduceLROnPlateau` — lowers learning rate on stagnation

---

## Project Structure

```
.
├── main.ipynb              # Main training notebook
├── final_dataset/          # Merged dataset (created at runtime)
│   ├── gir/
│   ├── jersey/
│   ├── jaffarabadi/
│   └── ...
└── final_dataset.zip       # Archived dataset (optional export)
```

---

## Requirements

```
tensorflow
kagglehub
numpy
pandas
matplotlib
seaborn
```

Install with:
```bash
pip install tensorflow kagglehub numpy pandas matplotlib seaborn
```

---

## Setup & Usage

### 1. Configure Kaggle API
Set your Kaggle API token in the notebook:
```python
os.environ["KAGGLE_API_TOKEN"] = "your_kaggle_api_token_here"
```

### 2. Run the notebook
Open `main.ipynb` in Google Colab (GPU recommended) and run all cells in order:
1. Install dependencies & set credentials
2. Download and merge datasets
3. Explore class distribution
4. Build and train the EfficientNetB0 model
5. Evaluate and visualize results

---

## Hardware

- Trained on **Google Colab T4 GPU**
- Recommended: GPU runtime for reasonable training times given the dataset size (~3 GB)

---

## Notes

- Duplicate filenames across the two datasets are handled automatically (appended with `_copy`).
- Breed folder names are normalized to lowercase with underscores (e.g., `Holstein Friesian` → `holstein_friesian`).
- Some breeds have very few samples (e.g., `kherigarh: 36`), which may affect per-class accuracy. Consider augmentation or oversampling for imbalanced classes.
