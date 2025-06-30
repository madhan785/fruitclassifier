# Fruit Image Classification

## Overview

This project implements an advanced image classification solution using a pretrained MobileNetV2 model to accurately classify images of fruits. The solution encompasses comprehensive data preprocessing, model training, and prediction capabilities.

## Project Structure

```
fruit-image-classification/
│
├── data/
│   ├── archive (6).zip
│   └── MY_data
|     ├── train/           # Training images
│     ├── test/            # Testing images
│     └── prediction/      # Images for prediction
│
├── src/
│   ├── preprocessing.py     # Dataset preprocessing and visualization
│   ├── model_training.py    # Model training script
│   ├── classify.py          # Prediction script
│   └── config.py            # Centralized configuration
│
├── results/              # Output storage
|   ├── plots/
|   |    └── sample_plot.png
│   ├── model_architecture.json
|   └── classifier_history.json
│
├── trained models/
│   └── fruit_classifier_model.h5
│
├── requirements.txt
└── README.md
```

## Key Features

* **Pretrained Model**: Leverages MobileNetV2 for robust fruit image classification
* **Advanced Preprocessing**: Comprehensive image normalization
* **Training Optimization**: 
  - Early stopping callback
  - Performance monitoring
* **Visualization Tools**: Dataset sample plotting
* **Flexible Prediction**: Real-time image classification script

## Installation

### Prerequisites

* Python 3.8+
* pip package manager

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Nirikshan95/FruitClassifier.git
   cd FruitClassifier
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Dataset**
   * Download "Fruit Classification (10 Class)" dataset from Kaggle
   * Place `archive (6).zip` in the `data/` directory
   * The preprocessing script will handle extraction

## Usage

### 1. Train the Model

```bash
python src/model_training.py
```

### 2. Predict New Images

```bash
python src/classify.py
```
*Note: Update `demo_img` path in `classify.py` with your image*

### 3. Visualize Dataset

```bash
python src/preprocessing.py
```

## Configuration

The `config.py` file centralizes project configurations:

```python
# Paths
DATA_DIR = "./data"
ZIP_FILE_PATH = "./data/archive (6).zip"
TRAINING_DATA = "./data/MY_data/train/"
TESTING_DATA = "./data/MY_data/test/"
PREDICTION_DATA = "./data/MY_data/prediction/"
MODEL_SAVE_PATH = "./trained models/fruit_classifier_model.h5"
MODEL_HISTORY_PATH = "./results/classifier_history.json"
MODEL_ARCHITECTURE_PATH="./results/model_architecture.json"
SAMPLE_PLOT_PATH="./results/plots/sample_plot.png"

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
OPTIMIZER = "adam"

# Model Parameters
INPUT_SHAPE = (224,224,3)
ACTIVATION_FUNCTION = "relu"
```

## Results

* **Trained Model**: `fruit_classifier_model.h5`
* **Model Architecture**: `results/model_architecture.json`
* **Performance Metrics**: 
  - Training accuracy
  - Validation accuracy

## Future Roadmap

- [ ] Expand fruit class diversity
- [ ] Fine-tune model architecture
- [ ] Implement advanced data augmentation
- [ ] Add model interpretability features

## References

1. [Kaggle Fruit Classification Dataset](https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class)
2. MobileNetV2 Research Paper

## Contributions

Contributions, issues, and feature requests are welcome!
