import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
from tqdm import tqdm
import faiss
from config import *

class FeatureExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_model = None
        self.text_model = None
        self.image_transform = None
        self.setup_models()
    
    def setup_models(self):
        """Initialize pre-trained models"""
        print(f"Setting up models on device: {self.device}")
        
        # Image feature extraction model (ResNet50)
        self.image_model = models.resnet50(pretrained=True)
        self.image_model.fc = nn.Identity()  
        self.image_model.to(self.device)
        self.image_model.eval()
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Text feature extraction model
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Models loaded successfully!")
    
    def extract_image_features(self, image_path):
        """Extract features from a single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.image_model(image_tensor)
            
            return features.cpu().numpy().flatten()
        
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return np.zeros(FEATURE_DIM)
    
    def extract_text_features(self, text):
        """Extract features from text"""
        try:
            features = self.text_model.encode(text)
            return features
        except Exception as e:
            print(f"Error processing text: {e}")
            return np.zeros(384)  # Default dimension for all-MiniLM-L6-v2
    
    def extract_all_features(self, data):
        """Extract features for all products"""
        print("Extracting image and text features...")
        
        image_features = []
        text_features = []
        valid_indices = []
        
        for idx, row in tqdm(data.iterrows(), total=len(data)):
            # Extract image features
            image_path = IMAGES_DIR / f"{row['product_id']}.jpg"
            
            if image_path.exists():
                img_feat = self.extract_image_features(image_path)
                txt_feat = self.extract_text_features(row['combined_text'])
                
                image_features.append(img_feat)
                text_features.append(txt_feat)
                valid_indices.append(idx)
            else:
                print(f"Image not found for product {row['product_id']}")
        
        image_features = np.array(image_features)
        text_features = np.array(text_features)
        
        print(f"Extracted features for {len(image_features)} products")
        print(f"Image features shape: {image_features.shape}")
        print(f"Text features shape: {text_features.shape}")
        
        return image_features, text_features, valid_indices
    
    def create_combined_features(self, image_features, text_features, alpha=0.7):
        """Combine image and text features with proper handling of edge cases"""
        print(f"Input shapes - Image: {image_features.shape}, Text: {text_features.shape}")
        
        # Handle NaN and zero values
        image_features = np.nan_to_num(image_features, nan=0.0, posinf=0.0, neginf=0.0)
        text_features = np.nan_to_num(text_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize features with epsilon to avoid division by zero
        epsilon = 1e-8
        
        image_norms = np.linalg.norm(image_features, axis=1, keepdims=True)
        image_norms = np.maximum(image_norms, epsilon)  
        image_features_norm = image_features / image_norms
        
        text_norms = np.linalg.norm(text_features, axis=1, keepdims=True)
        text_norms = np.maximum(text_norms, epsilon)  
        text_features_norm = text_features / text_norms
        
        # Concatenate features
        combined_features = np.concatenate([
            alpha * image_features_norm, 
            (1 - alpha) * text_features_norm
        ], axis=1)
        
        # Final normalization
        combined_norms = np.linalg.norm(combined_features, axis=1, keepdims=True)
        combined_norms = np.maximum(combined_norms, epsilon)
        combined_features = combined_features / combined_norms
        
        print(f"Combined features shape: {combined_features.shape}")
        print(f"Combined features stats - min: {combined_features.min():.6f}, max: {combined_features.max():.6f}")
        
        return combined_features


    def save_features(self, image_features, text_features, combined_features, valid_indices):
        """Save extracted features"""
        features_data = {
            'image_features': image_features,
            'text_features': text_features,
            'combined_features': combined_features,
            'valid_indices': valid_indices
        }
        
        with open(PROCESSED_DATA_DIR / 'features.pkl', 'wb') as f:
            pickle.dump(features_data, f)
        
        print("Features saved successfully!")
    
    def load_features(self):
        """Load saved features"""
        features_file = PROCESSED_DATA_DIR / 'features.pkl'
        if features_file.exists():
            with open(features_file, 'rb') as f:
                features_data = pickle.load(f)
            print("Features loaded successfully!")
            return features_data
        else:
            print("No saved features found.")
            return None

class FAISSIndexer:
    def __init__(self):
        self.index = None
        self.product_ids = None
    
    def build_index(self, features, product_ids):
        """Build FAISS index for fast similarity search"""
        print("Building FAISS index...")
        
        # Normalize features
        features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        # Create FAISS index
        dimension = features_norm.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(features_norm.astype('float32'))
        
        self.product_ids = product_ids
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, query_features, k=10):
        """Search for similar items"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Normalize query features
        query_norm = query_features / np.linalg.norm(query_features)
        query_norm = query_norm.reshape(1, -1).astype('float32')
        
        # Search
        scores, indices = self.index.search(query_norm, k)
        
        return scores[0], indices[0]
    
    def save_index(self, filepath):
        """Save FAISS index"""
        faiss.write_index(self.index, str(filepath))
        
        # Save product IDs
        with open(filepath.parent / 'product_ids.pkl', 'wb') as f:
            pickle.dump(self.product_ids, f)
        
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath):
        """Load FAISS index"""
        if Path(filepath).exists():
            self.index = faiss.read_index(str(filepath))
            
            # Load product IDs
            with open(filepath.parent / 'product_ids.pkl', 'rb') as f:
                self.product_ids = pickle.load(f)
            
            print(f"Index loaded from {filepath}")
            return True
        return False

if __name__ == "__main__":
    # Load processed data
    with open(PROCESSED_DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    
    # Extract features
    extractor = FeatureExtractor()
    image_features, text_features, valid_indices = extractor.extract_all_features(data)
    
    # Create combined features
    combined_features = extractor.create_combined_features(image_features, text_features)
    
    # Save features
    extractor.save_features(image_features, text_features, combined_features, valid_indices)
    
    # Build FAISS index
    indexer = FAISSIndexer()
    product_ids = [data.iloc[i]['product_id'] for i in valid_indices]
    indexer.build_index(combined_features, product_ids)
    indexer.save_index(PROCESSED_DATA_DIR / 'faiss_index.bin')
    
    print("Feature extraction and indexing completed!")
