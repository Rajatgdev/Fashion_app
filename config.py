import os
from pathlib import Path

# Base paths - Make sure BASE_DIR is a Path object
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
IMAGES_DIR = DATA_DIR / "images"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, IMAGES_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configurations
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
FEATURE_DIM = 2048
TOP_K_RESULTS = 20



# File paths
DRESSES_FILE = RAW_DATA_DIR / "dresses_bd_processed_data.csv"
JEANS_FILE = RAW_DATA_DIR / "jeans_bd_processed_data.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "combined_data.pkl"
FEATURES_FILE = PROCESSED_DATA_DIR / "image_features.npy"
FAISS_INDEX_FILE = PROCESSED_DATA_DIR / "faiss_index.bin"
