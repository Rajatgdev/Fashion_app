import pandas as pd
import numpy as np
import json
import ast
import requests
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import pickle
from config import *

class DataPreprocessor:
    def __init__(self):
        self.combined_data = None
        
    def load_data(self):
        """Load and combine dress and jeans data"""
        print("Loading datasets...")
        
        # Load datasets
        dresses_df = pd.read_csv(DRESSES_FILE)
        jeans_df = pd.read_csv(JEANS_FILE)
        
        # Add category labels
        dresses_df['category'] = 'dress'
        jeans_df['category'] = 'jeans'
        
        # Combine datasets
        self.combined_data = pd.concat([dresses_df, jeans_df], ignore_index=True)
        
        print(f"Total products loaded: {len(self.combined_data)}")
        print(f"Dresses: {len(dresses_df)}, Jeans: {len(jeans_df)}")
        return self.combined_data
    
    def clean_data(self):
        """Clean and preprocess data with multi-currency support"""
        print("Cleaning data...")
        
        def safe_eval(x):
            try:
                if isinstance(x, str):
                    return ast.literal_eval(x)
                return x
            except:
                return {}
        
        # Clean price columns
        self.combined_data['selling_price'] = self.combined_data['selling_price'].apply(safe_eval)
        self.combined_data['mrp'] = self.combined_data['mrp'].apply(safe_eval)
        
        # Enhanced price extraction with currency conversion
        def extract_price_with_conversion(price_dict):
            if isinstance(price_dict, dict):
                if 'INR' in price_dict:
                    inr_price = price_dict['INR']
                    if isinstance(inr_price, (int, float)) and inr_price > 0:
                        return inr_price
                
                if 'USD' in price_dict:
                    usd_price = price_dict['USD']
                    if isinstance(usd_price, (int, float)) and usd_price > 0:
                        return usd_price * 83  
                
                for currency, price in price_dict.items():
                    if isinstance(price, (int, float)) and price > 0:
                        if currency == 'EUR':
                            return price * 90  # EUR to INR
                        elif currency == 'GBP':
                            return price * 105  # GBP to INR
                        else:
                            return price * 83  
            
            # Handle direct numeric values
            if isinstance(price_dict, (int, float)) and price_dict > 0:
                return price_dict
            
            return 0
        
        # Apply enhanced price extraction
        self.combined_data['selling_price_inr'] = self.combined_data['selling_price'].apply(extract_price_with_conversion)
        self.combined_data['mrp_inr'] = self.combined_data['mrp'].apply(extract_price_with_conversion)
        
        # Debug: Check price distribution by category
        dress_prices = self.combined_data[self.combined_data['category'] == 'dress']['selling_price_inr']
        jeans_prices = self.combined_data[self.combined_data['category'] == 'jeans']['selling_price_inr']
        
        print(f"Dress prices > 0: {sum(dress_prices > 0)} out of {len(dress_prices)}")
        print(f"Jeans prices > 0: {sum(jeans_prices > 0)} out of {len(jeans_prices)}")
        print(f"Sample jeans prices after conversion: {jeans_prices.head(5).tolist()}")
        
        # Calculate discount properly
        def calculate_discount(row):
            if row['mrp_inr'] > 0 and row['selling_price_inr'] > 0:
                return ((row['mrp_inr'] - row['selling_price_inr']) / row['mrp_inr']) * 100
            return 0
        
        self.combined_data['discount'] = self.combined_data.apply(calculate_discount, axis=1)
        
        # Parse feature lists and image lists
        self.combined_data['feature_list'] = self.combined_data['feature_list'].apply(safe_eval)
        self.combined_data['pdp_images_s3'] = self.combined_data['pdp_images_s3'].apply(safe_eval)
        
        # Clean text fields
        text_columns = ['product_name', 'description', 'meta_info']
        for col in text_columns:
            self.combined_data[col] = self.combined_data[col].fillna('').astype(str)
        
        # Now filter out products with zero prices (after conversion)
        initial_count = len(self.combined_data)
        self.combined_data = self.combined_data[self.combined_data['selling_price_inr'] > 0]
        
        # Remove rows with missing essential data
        self.combined_data = self.combined_data.dropna(subset=['feature_image_s3', 'product_id'])
        
        print(f"Data cleaned. Removed {initial_count - len(self.combined_data)} products with zero prices")
        print(f"Remaining products: {len(self.combined_data)}")
        print(f"Products by category after cleaning:")
        print(self.combined_data['category'].value_counts())
        
        return self.combined_data

    def run_preprocessing(self, max_images=7000):
        """Run preprocessing with balanced category sampling"""
        print("Starting data preprocessing pipeline...")
        
        # Load and clean data
        self.load_data()
        self.clean_data()
        
        # Get balanced samples from both categories
        dresses = self.combined_data[self.combined_data['category'] == 'dress']
        jeans = self.combined_data[self.combined_data['category'] == 'jeans']
        
        print(f"Available dresses with prices: {len(dresses)}")
        print(f"Available jeans with prices: {len(jeans)}")
        
        # Take balanced samples (50% each category)
        dress_sample = dresses.head(max_images // 2)
        jeans_sample = jeans.head(max_images // 2)
        
        # Combine balanced samples
        self.combined_data = pd.concat([dress_sample, jeans_sample], ignore_index=True)
        
        print(f"Processing {len(dress_sample)} dresses and {len(jeans_sample)} jeans")

        self.download_images(max_images)  

        self.create_combined_features()

        self.save_processed_data()
        
        print("Preprocessing completed!")
        return self.combined_data

    
    def download_images(self, max_images=7000):
        """Download product images for processing"""
        print(f"Downloading up to {max_images} images...")
        
        # Create images directory
        IMAGES_DIR.mkdir(exist_ok=True)
        
        downloaded_count = 0
        failed_downloads = []
        
        for idx, row in tqdm(self.combined_data.head(max_images).iterrows(), total=min(max_images, len(self.combined_data))):
            try:
                image_url = row['feature_image_s3']
                product_id = row['product_id']
                
                # Skip if already downloaded
                image_path = IMAGES_DIR / f"{product_id}.jpg"
                if image_path.exists():
                    downloaded_count += 1
                    continue
                
                # Download image with better headers
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(image_url, timeout=30, headers=headers)
                response.raise_for_status()
                
                # Save image
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                
                # Verify image can be opened
                Image.open(image_path).verify()
                downloaded_count += 1
                
            except Exception as e:
                failed_downloads.append((idx, str(e)))
                continue
        
        print(f"Successfully downloaded {downloaded_count} images")
        print(f"Failed downloads: {len(failed_downloads)}")
        
        return downloaded_count, failed_downloads
    
    def create_combined_features(self):
        """Create combined text features from multiple columns"""
        print("Creating combined text features...")
        
        def combine_text_features(row):
            features = []
            
            # Add product name
            if row['product_name']:
                features.append(row['product_name'])
            
            # Add brand
            if row['brand']:
                features.append(row['brand'])
            
            # Add category
            features.append(row['category'])
            
            # Add description
            if row['description']:
                features.append(row['description'])
            
            # Add feature list
            if isinstance(row['feature_list'], list):
                features.extend(row['feature_list'])
            
            # Add meta info
            if row['meta_info']:
                features.append(row['meta_info'])
            
            return ' '.join(features)
        
        self.combined_data['combined_text'] = self.combined_data.apply(combine_text_features, axis=1)
        
        return self.combined_data
    
    def save_processed_data(self):
        """Save processed data"""
        print("Saving processed data...")
        
        with open(PROCESSED_DATA_FILE, 'wb') as f:
            pickle.dump(self.combined_data, f)
        
        print(f"Processed data saved to {PROCESSED_DATA_FILE}")
    
    def load_processed_data(self):
        """Load processed data"""
        if PROCESSED_DATA_FILE.exists():
            with open(PROCESSED_DATA_FILE, 'rb') as f:
                self.combined_data = pickle.load(f)
            print(f"Loaded processed data: {len(self.combined_data)} products")
            return self.combined_data
        else:
            print("No processed data found. Please run preprocessing first.")
            return None
    
    def run_preprocessing(self, max_images=7000):  # Increased to get more jeans
        """Run complete preprocessing pipeline with balanced categories"""
        print("Starting data preprocessing pipeline...")
        
        # Load and clean data
        self.load_data()
        self.clean_data()
        
        # Get balanced samples from both categories
        dresses = self.combined_data[self.combined_data['category'] == 'dress']
        jeans = self.combined_data[self.combined_data['category'] == 'jeans']
        
        print(f"Available dresses: {len(dresses)}")
        print(f"Available jeans: {len(jeans)}")
        
        # Take balanced samples
        dress_sample = dresses.head(max_images // 2)
        jeans_sample = jeans.head(max_images // 2)
        
        # Combine balanced samples
        self.combined_data = pd.concat([dress_sample, jeans_sample], ignore_index=True)
        
        print(f"Processing {len(dress_sample)} dresses and {len(jeans_sample)} jeans")
        
        self.download_images(max_images)
        
        self.create_combined_features()
        
        self.save_processed_data()
        
        print("Preprocessing completed!")
        return self.combined_data


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    data = preprocessor.run_preprocessing(max_images=7000)  
