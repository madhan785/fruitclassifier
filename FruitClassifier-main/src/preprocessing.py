import os
import zipfile
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.callbacks import EarlyStopping
import config

class DataLoader:
    """
    A class to handle dataset loading, preprocessing, and visualization.
    """
    def __init__(self):
        self.classes = []
        self.train_data = None
        self.test_data = None

    def load_data(self):
        """
        Load and preprocess the dataset. Extract files if necessary.
        """
        default_path = config.DATA_DIR

        # Check if the dataset exists
        if os.path.exists(default_path) and os.listdir(default_path):
            print(f"Dataset already exists at: {default_path}")
            print("Files in target directory:", os.listdir(default_path))
        else:
            print("Download dataset from: 'https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class'")
            
            # Ensure the target directory exists
            if not os.path.exists(default_path):
                os.makedirs(default_path, exist_ok=True)
            if not os.listdir(default_path):
                raise FileNotFoundError(
                    f"Error: No files found in {default_path}. "
                    f"Insert the downloaded zipfile in the folder {default_path}."
                )

        # Unzip the dataset
        try:
            with zipfile.ZipFile(config.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(config.DATA_DIR)
        except zipfile.BadZipFile:
            raise ValueError(f"Error: Unable to unzip {config.ZIP_FILE_PATH}. Check the file format.")
        
        # Load training and testing datasets
        self.train_data_org = tf.keras.utils.image_dataset_from_directory(
            config.TRAINING_DATA,
            shuffle=True,
            image_size=(224, 224),
            batch_size=30,
            validation_split=False
        )
        self.test_data_org = tf.keras.utils.image_dataset_from_directory(
            config.TESTING_DATA,
            shuffle=True,
            image_size=(224, 224),
            batch_size=30,
            validation_split=False
        )

        self.classes = self.train_data_org.class_names
        
        #rescaling 
        def rescale(image,label):
            rescaler=Rescaling(1./255)
            image=rescaler(image)
            return image,label
        self.train_data=self.train_data_org.map(rescale)
        self.test_data=self.test_data_org.map(rescale)

    def plot_samples(self):
        """
        Plot a few sample images from the training dataset.
        """
        if self.train_data is None:
            raise ValueError("Train data is not loaded. Call `load_data()` first.")

        print('Plotting sample images...')
        if not os.path.exists('./results/plots/'):
            os.makedirs('./results/plots/')
        plt.figure(figsize=(10, 10))
        for images, labels in self.train_data_org.take(1):
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype('uint8'))
                plt.title(self.classes[labels[i]])
                plt.axis('off')
        plt.savefig(config.SAMPLE_PLOT_PATH)
        plt.show()
        

    def get_classes(self):
        """
        Get the list of class names.
        """
        return self.classes

class call_back:
    """
    A class to handle callbacks.
    """
    def __init__(self):
        self.callback=None
    
    def get_callbacks(self):
        self.callback=EarlyStopping(
            monitor="val_loss",
            min_delta=0.2,
            patience=3,
            verbose=1,
            mode="auto",
            restore_best_weights=True
        )
        return self.callback

# Entry Point
if __name__ == "__main__":
    data_loader = DataLoader()
    
    # Load the dataset
    data_loader.load_data()

    # Plot sample images
    data_loader.plot_samples()