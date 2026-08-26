# Image Object Recognition — MobileNetV2 Transfer Learning

Multi-class image classification on the ImageNet-small dataset using transfer
learning with MobileNetV2 (ImageNet-pretrained, frozen base + trained
classification head).

## Setup Instructions

### Dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn
```

Tested with:
- Python 3.12
- TensorFlow / Keras (includes `tensorflow.keras.applications.MobileNetV2`)
- numpy, matplotlib
- scikit-learn (for `confusion_matrix`, `classification_report`)
- seaborn (for the confusion matrix heatmap)

### Dataset

Uses the **ImageNet-small dataset** (Kaggle: `mdjahed0/imagenet-small-dataset-for-alexnet`),
organized in the standard `flow_from_directory` format:

```
training_set/
  class_1/
    image1.jpg
    image2.jpg
    ...
  class_2/
    ...
test_set/
  class_1/
    ...
  class_2/
    ...
```

If running on Kaggle, attach the dataset via **Add Data** and the notebook's
`train_path` / `test_path` variables will resolve automatically. If running
elsewhere, update `train_path` and `test_path` at the top of the data-loading
cell to point at your local copy of the same folder structure.

### How to Run

1. Open `image-object-recognition-mobilenetv2.ipynb` in Jupyter, Kaggle, or
   Google Colab.
2. **Enable GPU** (Kaggle: Settings → Accelerator → GPU; Colab: Runtime →
   Change runtime type → GPU). Training on CPU will be much slower.
3. Run all cells top to bottom. The notebook will:
   - Load and augment the dataset
   - Build the MobileNetV2-based model
   - Train with `EarlyStopping` + `ReduceLROnPlateau`
   - Evaluate on the held-out test set
   - Run a sample prediction on one test image

## Project Details

### Dataset

- **Source**: ImageNet-small dataset, split into `training_set/` and `test_set/`
  folders, one subfolder per class.
- **Split**: 80% training / 20% validation, carved out of `training_set/` via
  `ImageDataGenerator(validation_split=0.2)`. `test_set/` is held out entirely
  and only used for final evaluation.

### Preprocessing / Augmentation

- Input images resized to **224×224** (MobileNetV2's expected input size).
- Training data augmentation via `ImageDataGenerator`:
  - Rotation (±20°)
  - Width/height shift (±15%)
  - Shear (10%)
  - Zoom (±15%)
  - Horizontal flip
  - `fill_mode='nearest'`
- All splits (train/val/test) use MobileNetV2's `preprocess_input` for pixel
  normalization, matching what the pretrained base expects.

### Model Architecture

Transfer learning on top of **MobileNetV2** (ImageNet weights, `include_top=False`):

```
MobileNetV2 base (frozen, ImageNet weights)
  → GlobalAveragePooling2D
  → Dense(256, activation='relu')
  → Dropout(0.5)
  → Dense(num_classes, activation='softmax')
```

The base network is frozen — only the small classification head is trained.
This keeps the number of trainable parameters low, which trains much faster
and generalizes better on a small dataset than training a deep CNN from
scratch.

### Training Approach

- **Optimizer**: Adam
- **Loss**: categorical cross-entropy
- **Epochs**: up to 105, with:
  - `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)`
- Training and validation accuracy/loss are tracked and plotted per epoch.

### Evaluation

- Test-set loss and accuracy via `model.evaluate()`.
- Full `classification_report` (precision/recall/F1 per class).
- Confusion matrix, visualized as a heatmap.
- A qualitative sample prediction: one test image is loaded, run through the
  trained model, and shown with its predicted class and confidence score.

## Output Preview

> Run the notebook and drop the corresponding screenshots/images here so the
> project is understandable at a glance.

- **Training curves** — accuracy and loss over epochs (train vs. validation):
  `![Accuracy/Loss curves](path/to/accuracy_loss_curves.png)`
- **Confusion matrix** on the test set:
  `![Confusion matrix](path/to/confusion_matrix.png)`
- **Sample prediction** — test image with predicted class and confidence:
  `![Sample prediction](path/to/sample_prediction.png)`
- **Test accuracy / loss**: fill in after running, e.g. `Test Accuracy: 0.XX`,
  `Test Loss: 0.XX`.
