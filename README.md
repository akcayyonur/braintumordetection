# Brain Tumor MRI Classification System

A deep learning-based medical imaging application for automated classification of brain tumors using MRI scans. This system uses **MobileNetV2** transfer learning to classify brain tumors into four categories: No Tumor, Meningioma, Glioma, and Pituitary. Built with **Streamlit** for an intuitive web interface.

## 🧠 Features

- **Multi-class Classification**: Detects and classifies 4 tumor types
  - No Tumor (healthy)
  - Meningioma
  - Glioma
  - Pituitary
- **Pre-trained Transfer Learning**: MobileNetV2 backbone with custom head
- **Two-Stage Training**: Head training + fine-tuning for optimal performance
- **Class Balancing**: Automatic class weight computation for imbalanced datasets
- **Domain-Robust Augmentation**: 8+ augmentation techniques for generalization
- **Web Interface**: User-friendly Streamlit dashboard for predictions
- **Confidence Thresholds**: Class-specific thresholds for reliable predictions
- **Fast Inference**: MobileNetV2 enables real-time predictions on CPU

## 📊 Model Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 (ImageNet pre-trained)
    ↓
Global Average Pooling
    ↓
Dense(256, ReLU) + Dropout(0.45)
    ↓
Output (4 classes, Softmax)
```

### Model Specifications

| Aspect | Details |
|--------|---------|
| **Base Model** | MobileNetV2 (ImageNet weights) |
| **Input Size** | 224×224×3 RGB images |
| **Output Classes** | 4 (no_tumor, meningioma, glioma, pituitary) |
| **Parameters** | ~3.5M (trainable during fine-tuning) |
| **Framework** | TensorFlow/Keras 2.15 |
| **Inference Time** | ~50-100ms per image |

## 🚀 Quick Start

### Prerequisites

```
Python >= 3.9
pip >= 21.0
CUDA 12.0+ (optional, for GPU acceleration)
```

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd brain-tumor-classification
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```


### Training the Model

```bash
python src/train_multiclass.py
```

This will:
1. Load MobileNetV2 with ImageNet weights
2. Train the classification head (Stage 1)
3. Fine-tune the last 60 layers of backbone (Stage 2)
4. Save the model to `models/brain_tumor_multiclass.h5`
5. Save metadata to `models/meta.json`
6. Display confusion matrix and classification report

### Running the Web Application

```bash
streamlit run src/app.py
```

Then open your browser to `http://localhost:8501`

### Model Evaluation

```bash
python src/evaluate.py
```

This computes:
- AUC-ROC score on validation set
- Optimal thresholds using Youden's J statistic
- Saves thresholds to `models/thresholds.json`


## 🔧 Configuration

### Training Parameters

Edit `src/train_multiclass.py` to customize:

```python
# Image preprocessing
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Stage 1: Head training
EPOCHS_STAGE1 = 12
LR_STAGE1 = 1e-4

# Stage 2: Fine-tuning
EPOCHS_STAGE2 = 15
LR_STAGE2 = 5e-6

# Backbone: fine-tune last N layers
UNFREEZE_LAST_N_LAYERS = 60  # Critical for domain shift handling

# Class weights (automatic)
class_weights = compute_class_weight(...)
```

### Prediction Thresholds

Customize class-specific confidence thresholds in `src/app.py`:

```python
THRESHOLDS = {
    "no_tumor": 0.85,      # Higher threshold for healthy cases
    "meningioma": 0.60,    # Moderate threshold
    "glioma": 0.55,        # Moderate threshold
    "pituitary": 0.55      # Moderate threshold
}
```

## 🧠 Training Pipeline Details

### Stage 1: Head Training (12 epochs)

1. **Base Model Frozen**: MobileNetV2 weights remain fixed (ImageNet features)
2. **Head Training**: Only the custom classification head is trained
3. **Purpose**: Quickly adapt pre-trained features to tumor classification task
4. **Learning Rate**: 1e-4 (relatively high for head training)
5. **Data Augmentation**: Applied to training data
6. **Early Stopping**: Patience=4 on validation loss

### Stage 2: Fine-Tuning (15 epochs)

1. **Backbone Unfrozen**: Last 60 layers of MobileNetV2 become trainable
2. **Purpose**: Adapt deep features to domain-specific characteristics
3. **Learning Rate**: 5e-6 (much lower for fine-tuning)
4. **Batch Normalization**: Layers retain learned statistics
5. **Early Stopping**: Patience=5 on validation loss
6. **Progressive Improvement**: LR scheduler reduces LR on plateau

### Data Augmentation Strategy

Robust augmentation to handle dataset variations:

```python
ImageDataGenerator(
    rotation_range=20,           # Random rotations (±20°)
    zoom_range=0.25,            # Random zoom (0.75x - 1.25x)
    width_shift_range=0.08,     # Horizontal shift (8%)
    height_shift_range=0.08,    # Vertical shift (8%)
    shear_range=0.10,           # Shear transformations
    brightness_range=(0.7, 1.3), # Brightness variation
    horizontal_flip=True        # Random horizontal flips
)
```

## 🎯 Web Interface (Streamlit)

### Features

- **Patient Information Input**: Name and scan date
- **Image Upload**: Drag-and-drop or browse
- **Real-Time Prediction**: Instant classification
- **Visual Feedback**: Color-coded results
  - 🟢 Green: No Tumor (confident)
  - 🔴 Red: Tumor (confident)
  - ⚠️ Yellow: Ambiguous (recommend doctor consultation)
- **Detailed Probabilities**: Expandable section showing all class probabilities
- **Medical-Grade Display**: Clear, professional interface

### Usage Flow

1. Enter patient name and scan date
2. Upload MRI image (JPG, PNG, JPEG)
3. System processes image and displays:
   - Main classification result
   - Confidence percentage
   - Recommendation
4. Expand "Detail Probabilities" to see all predictions
5. Print or save results as needed

## 📈 Model Evaluation Metrics

### Validation Performance

- **Accuracy**: Percentage of correct classifications
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

### Confusion Matrix

Shows classification performance across all four classes:

```
              Predicted
            T0 T1 T2 T3
Actual T0 [ ]  
       T1    [ ]
       T2       [ ]
       T3          [ ]
```

T0 = no_tumor, T1 = meningioma, T2 = glioma, T3 = pituitary

## 🔍 Prediction Logic

### Two-Stage Decision Making

**Step 1: Is there a tumor?**
```
if P(no_tumor) >= 0.85:
    → "No Tumor" (high confidence)
else:
    → Go to Step 2
```

**Step 2: What type of tumor?**
```
tumor_type = argmax(P(meningioma), P(glioma), P(pituitary))

if P(tumor_type) >= class_threshold:
    → "[TUMOR_TYPE] Tumor" (high confidence)
else:
    → "Ambiguous" (recommend doctor)
```

### Confidence Interpretation

| Confidence | Recommendation |
|-----------|-----------------|
| **≥ 0.85** | High confidence, likely accurate |
| **0.60-0.84** | Moderate confidence, review recommended |
| **< 0.60** | Low confidence, consult physician |

## ⚠️ Important Notes

### Medical Disclaimer

⚠️ **This system is for research and educational purposes only.**

- **NOT a diagnostic tool**: Cannot be used for medical diagnosis
- **Not a substitute**: Professional radiologist review is required
- **Experimental**: Results should be verified by medical experts
- **Regulatory**: Not approved by FDA or medical authorities

### Data Handling

- **Patient Privacy**: Implement HIPAA-compliant data handling
- **Encryption**: Use HTTPS for data transmission
- **Logging**: Minimize personal information logging
- **Access Control**: Restrict model access to authorized users

### Model Limitations

1. **Domain Shift**: Performance varies with scanner/protocol differences
2. **Input Quality**: Requires good quality, clear MRI images
3. **Resolution**: Expects standard 224×224 RGB format
4. **Artifacts**: May fail on images with significant motion/artifacts
5. **Edge Cases**: Untested on rare tumor subtypes


## 📊 Dataset Requirements

### Minimum Dataset Size

| Training Data | Expected Accuracy |
|--------------|------------------|
| < 100 images | Poor (<75%) |
| 100-500 images | Fair (75-85%) |
| 500-2000 images | Good (85-90%) |
| 2000+ images | Excellent (90%+) |

### Image Quality Standards

- **Format**: JPG, PNG (RGB or grayscale)
- **Size**: Preferably 256×256 or larger
- **Contrast**: Clear distinction between tumor and healthy tissue
- **Artifacts**: Minimal motion/scan artifacts
- **Annotation**: Accurate, consistent labeling

### Public Datasets

- [Kaggle: Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [BRATS Challenge](https://www.med.upenn.edu/cbica/brats2023/)
- [IXI Dataset](https://brain-development.org/ixi-dataset/)

## 🚀 Deployment Options

### Local Deployment

```bash
streamlit run src/app.py
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py"]
```

```bash
docker build -t brain-tumor-classifier .
docker run -p 8501:8501 brain-tumor-classifier
```

## 📚 References

### Medical Imaging
- [MRI Physics and Signal Processing](https://www.ncbi.nlm.nih.gov/books/NBK557556/)
- [Brain Tumor Classification](https://www.cancer.gov/types/brain/patient/brain-tumor-treatment-option-pdq)
- [Radiomics in Oncology](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5830500/)

### Deep Learning
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Transfer Learning in Medical Imaging](https://arxiv.org/abs/1805.03677)
- [Understanding Fine-tuning](https://cs231n.github.io/transfer-learning/)

### Tools & Frameworks
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn](https://scikit-learn.org/stable/)

### ⚖️ Legal Notice

This software is provided "as-is" for research and educational purposes. The authors and contributors are not responsible for any medical decisions made based on this system. Always consult qualified medical professionals for diagnosis and treatment.
