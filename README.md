# Emotion Recognition Dataset

**Public, open-source dataset for facial emotion recognition**

This repository provides a large-scale dataset for **emotion recognition** containing facial images labeled across four emotion categories: **Fear**, **Happy**, **Sad**, and **Neutral**. The dataset is freely available for machine learning and AI research, including deep learning model training, evaluation, and benchmarking.

---

## Dataset Overview

- **Total Images:** 19,000+
- **Classes (4):**
  - `fear/`
  - `happy/`
  - `sad/`
  - `neutral/`
- **File Format:** JPG or PNG
- **Average Image Size:** 48×48 or higher (varies)
- **Color Mode:** RGB

---

## Directory Structure

```
emotion-dataset/
├─ fear/
│  ├─ img_0001.jpg
│  ├─ img_0002.jpg
│  └─ ...
├─ happy/
│  ├─ img_0001.jpg
│  ├─ img_0002.jpg
│  └─ ...
├─ sad/
│  ├─ img_0001.jpg
│  ├─ img_0002.jpg
│  └─ ...
├─ neutral/
│  ├─ img_0001.jpg
│  ├─ img_0002.jpg
│  └─ ...
└─ README.md
```

Each folder contains all images belonging to a specific emotion class.

---

## Data Description

Each image contains a single human face expressing one of the four labeled emotions. The dataset is intended for use in:

- Emotion classification
- Facial expression recognition
- Affective computing
- Human–computer interaction

### Emotion Labels

| Label ID | Emotion | Folder Name |
|-----------|----------|--------------|
| 0 | Fear | `fear/` |
| 1 | Happy | `happy/` |
| 2 | Sad | `sad/` |
| 3 | Neutral | `neutral/` |

These labels can be used for categorical emotion classification tasks.

---

## Example Usage (Python)

### Load dataset using PyTorch

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root='emotion-dataset', transform=transform)

print(f"Total images: {len(dataset)}")
print(f"Classes: {dataset.classes}")
```

### Load dataset using TensorFlow / Keras

```python
import tensorflow as tf

img_height, img_width = 48, 48
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    'emotion-dataset',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print(class_names)
```

---

## Suggested Data Splits

Although this dataset is provided as a single collection, we recommend splitting it for training and evaluation:

| Split | Percentage | Approx. Images |
|--------|-------------|----------------|
| Train | 70% | ~13,300 |
| Validation | 15% | ~2,850 |
| Test | 15% | ~2,850 |

You can create these splits using tools like `scikit-learn` or `torch.utils.data.random_split`.

---

## Preprocessing Recommendations

- **Face alignment / cropping:** Ensure the face region is centered.
- **Grayscale conversion:** Optional, depending on model architecture.
- **Normalization:** Scale pixel values to `[0,1]` or `[-1,1]`.
- **Augmentation:** Random rotation, flips, color jitter for robustness.

---

## Example Model Training (Keras)

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(48, 48, 3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

---

## License

This dataset is distributed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

You are free to:
- Share — copy and redistribute the material
- Adapt — remix, transform, and build upon the material for any purpose

**Attribution:** Please credit the original dataset maintainer when using this data.

---

## Contact & Contributions

For contributions, bug reports, or suggestions:
- Open a GitHub Issue in this repository.
- Pull requests are welcome.

**Maintainer:** [Rohan Patil]  
**Year:** 2025

---

Thank you for using the Emotion Recognition Dataset! 😊

