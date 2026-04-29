# Indian Cattle & Buffalo Breed Classifier

A deep learning pipeline to classify **15 Indian cattle and buffalo breeds** from images using **YOLOv8** for object detection and **EfficientNetB0** for transfer learning-based classification.

---

## Overview

This project combines two Kaggle datasets of Indian bovine breeds, applies YOLO-based cropping to isolate animals, and trains a fine-tuned EfficientNetB0 model to identify breeds with ~64.7% validation accuracy.

---

## Supported Breeds

| Cattle Breeds | Buffalo Breeds |
|---|---|
| Gir, Sahiwal, Ongole | Murrah, Nagpuri |
| Kankrej, Tharparkar | Jaffarabadi, Nili-Ravi |
| Hallikar | Surti, Mehsana |
| Jersey, Holstein Friesian, Brown Swiss | |

---

## Dataset

Two Kaggle datasets are merged:

- [`lukex9442/indian-bovine-breeds`](https://www.kaggle.com/datasets/lukex9442/indian-bovine-breeds) — Indian cattle breeds
- [`atharvadarpude/indian-buffalo-dataset`](https://www.kaggle.com/datasets/atharvadarpude/indian-buffalo-dataset) — Indian buffalo breeds

**Total images after filtering & YOLO cropping:** ~4,106 across 15 classes

---

## Pipeline

```
Raw Datasets (Kaggle)
        ↓
   Merge & Normalize
        ↓
  Filter 15 Breeds
        ↓
  YOLOv8 Cropping        ← isolates the animal in each image
        ↓
  Train / Val Split       ← 80/20
        ↓
  Data Augmentation       ← rotation, zoom, horizontal flip
        ↓
  EfficientNetB0          ← frozen base, custom head
  (Transfer Learning)
        ↓
    Fine Tuning           ← last 15 layers unfrozen
        ↓
  Saved Model (.keras)
```

---

## Model Architecture

- **Base:** EfficientNetB0 (pretrained on ImageNet, top removed)
- **Head:**
  - GlobalAveragePooling2D
  - Dense(512, ReLU) → Dropout(0.5)
  - Dense(256, ReLU) → Dropout(0.3)
  - Dense(15, Softmax)
- **Input size:** 224×224×3

### Training Strategy

| Phase | Layers Trainable | LR | Epochs |
|---|---|---|---|
| Transfer Learning | Head only | 1e-4 | Up to 20 (early stop) |
| Fine Tuning | Last 15 base layers | 1e-5 | 5 |

**Callbacks:** `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`

---

## Results

| Split | Accuracy |
|---|---|
| Training | ~73.96% |
| Validation | ~64.73% |

---

## Getting Started

### Prerequisites

```bash
pip install tensorflow ultralytics opencv-python kagglehub scikit-learn matplotlib seaborn
```

### 1. Download Datasets

```python
import kagglehub
dataset1 = kagglehub.dataset_download("lukex9442/indian-bovine-breeds")
dataset2 = kagglehub.dataset_download("atharvadarpude/indian-buffalo-dataset")
```

### 2. Run the Notebook

Open `main.ipynb` in Google Colab or Jupyter and run all cells in order.

### 3. Predict on a New Image

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np

model = load_model("cattle_breed_model.keras")

labels = [
    'brown_swiss', 'gir', 'hallikar', 'holstein_friesian', 'jaffarabadi',
    'jersey', 'kankrej', 'mehsana', 'murrah', 'nagpuri',
    'nili_ravi', 'ongole', 'sahiwal', 'surti', 'tharparkar'
]

img = image.load_img("your_image.jpg", target_size=(224, 224))
img_array = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))

pred = model.predict(img_array)
print("Predicted Breed:", labels[np.argmax(pred)])
```

---

## Project Structure during runtime

```
├── main.ipynb                  # Main training notebook
├── cattle_breed_model.keras    # Saved model
├── final_dataset/              # Merged raw dataset (48 breeds)
├── filtered_dataset/           # 15 selected breeds
├── cropped_dataset/            # YOLO-cropped images
└── data/
    ├── train/                  # Training split (80%)
    └── val/                    # Validation split (20%)
```

---

## Tech Stack

- **Python** 3.x
- **TensorFlow / Keras** — model training
- **YOLOv8 (Ultralytics)** — animal detection & cropping
- **OpenCV** — image processing
- **scikit-learn** — train/val split
- **Kaggle Hub** — dataset download

---

## Future Improvements

- Add more breeds and larger datasets
- Implement Grad-CAM visualizations for explainability
- Deploy as a web app (Gradio / Streamlit)
- Try Vision Transformer (ViT) for higher accuracy
- Add confidence score thresholding for uncertain predictions

---

## License

This project is open-source and available under the [MIT License](LICENSE).
