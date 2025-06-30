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