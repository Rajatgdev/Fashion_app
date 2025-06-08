import numpy as np
import pandas as pd
from PIL import Image
import pickle
from pathlib import Path
from feature_extraction import FeatureExtractor, FAISSIndexer
from metadata_extractor import MetadataExtractor
from color_analyzer import ColorAnalyzer
from config import *

class VisualSearchEngine:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.indexer = FAISSIndexer()
        self.metadata_extractor = MetadataExtractor()
        self.color_analyzer = ColorAnalyzer()
        self.data = None
        self.features_data = None
        self.load_models_and_data()
    
    def load_models_and_data(self):
        print("Loading visual search engine...")
        
        # Load processed data
        try:
            with open(PROCESSED_DATA_FILE, 'rb') as f:
                self.data = pickle.load(f)
            print(f"Loaded product data: {len(self.data)} products")
        except FileNotFoundError:
            print("Processed data not found. Please run preprocessing first.")
            return False
        
        # Load features
        self.features_data = self.feature_extractor.load_features()
        if self.features_data is None:
            print("Features not found. Please run feature extraction first.")
            return False
        
        # Load FAISS index
        index_loaded = self.indexer.load_index(PROCESSED_DATA_DIR / 'faiss_index.bin')
        if not index_loaded:
            print("FAISS index not found. Please build index first.")
            return False
        
        print("Visual search engine loaded successfully!")
        return True
    

    def search_by_image_with_metadata(self, image_path, top_k=10):
        try:
            extractor = MetadataExtractor()
            metadata = extractor.extract_metadata(image_path)
            
            print(f"🎯 YOLOv8 detected category: {metadata['predicted_category']} ")
            
            # Get visual similarity results
            similar_products = self.search_by_image(image_path, top_k)
            
            return similar_products, metadata
            
        except Exception as e:
            print(f"❌ Error in enhanced search: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to basic search
            similar_products = self.search_by_image(image_path, top_k)
            metadata = {
                'predicted_category': 'dress', 
                'confidence': 0.5,
                'detection_method': 'fallback'
            }
            return similar_products, metadata

    
    def search_by_image(self, image_path, top_k=10):
        try:
            print(f"Starting image search for: {image_path}")
            
            query_filename = Path(image_path).name
            
            if not Path(image_path).exists():
                return []
            
            # Extract features
            query_image_features = self.feature_extractor.extract_image_features(image_path)
            
            if query_image_features is None or query_image_features.size == 0:
                return []
            
            query_image_features = np.nan_to_num(query_image_features, nan=0.0, posinf=0.0, neginf=0.0)
            dummy_text_features = np.zeros(384)
            
            combined_query_features = self.feature_extractor.create_combined_features(
                query_image_features.reshape(1, -1), 
                dummy_text_features.reshape(1, -1),
                alpha=0.95  # Even higher weight on visual features
            )
            
            query_features = combined_query_features.flatten()
            
            if np.any(np.isnan(query_features)) or np.any(np.isinf(query_features)):
                return []
            
            # Search using FAISS
            scores, indices = self.indexer.search(query_features, top_k * 3)
            
            # Enhanced similarity scoring
            enhanced_results = []
            
            for score, idx in zip(scores, indices):
                if idx != -1 and idx < len(self.features_data['valid_indices']):
                    data_idx = self.features_data['valid_indices'][idx]
                    product = self.data.iloc[data_idx]
                    
                    # Check for exact match
                    product_image_name = f"{product['product_id']}.jpg"
                    is_exact_match = (query_filename.find(product['product_id']) != -1 or 
                                    product_image_name in query_filename)
                    
                    if is_exact_match:
                        final_score = 1.0  # 100% for exact match
                    else:
                        # Better normalization for non-exact matches
                        # Map FAISS scores to 30-95% range
                        normalized_score = max(0.3, min(0.95, (score + 1) / 2))
                        final_score = normalized_score
                    
                    enhanced_results.append({
                        'score': final_score,
                        'idx': idx,
                        'is_exact': is_exact_match
                    })
            
            # Sort by score and exact match priority
            enhanced_results.sort(key=lambda x: (x['is_exact'], x['score']), reverse=True)
            
            # Get final results
            final_scores = [r['score'] for r in enhanced_results[:top_k]]
            final_indices = [r['idx'] for r in enhanced_results[:top_k]]
            
            results = self.get_search_results(final_scores, final_indices)
            
            return results
            
        except Exception as e:
            print(f"Error in image search: {e}")
            return []

    
    def get_search_results(self, scores, indices):
        results = []
        valid_indices = self.features_data['valid_indices']
        
        MIN_SIMILARITY_THRESHOLD = 0.20  # Only show results above 20% similarity
        
        for score, idx in zip(scores, indices):
            if idx < len(valid_indices):
                data_idx = valid_indices[idx]
                product = self.data.iloc[data_idx]
                
                # Check if this product has an image and meets threshold
                image_path = f"data/images/{product['product_id']}.jpg"
                
                if Path(image_path).exists() and score >= MIN_SIMILARITY_THRESHOLD:
                    result = {
                        'product_id': product['product_id'],
                        'product_name': product['product_name'],
                        'brand': product['brand'],
                        'category': product['category'],
                        'selling_price_inr': product['selling_price_inr'],
                        'mrp_inr': product['mrp_inr'],
                        'discount': product['discount'],
                        'image_url': product['feature_image_s3'],
                        'pdp_url': product['pdp_url'],
                        'description': product['description'][:200] + '...' if len(product['description']) > 200 else product['description'],
                        'similarity_score': score
                    }
                    results.append(result)
        
        return results
    
    def get_product_by_id(self, product_id):
        """Get detailed product information by ID"""
        product_row = self.data[self.data['product_id'] == product_id]
        
        if len(product_row) == 0:
            return None
        
        product = product_row.iloc[0]
        
        return {
            'product_id': product['product_id'],
            'product_name': product['product_name'],
            'brand': product['brand'],
            'category': product['category'],
            'selling_price_inr': product['selling_price_inr'],
            'mrp_inr': product['mrp_inr'],
            'discount': product['discount'],
            'image_url': product['feature_image_s3'],
            'additional_images': product['pdp_images_s3'] if isinstance(product['pdp_images_s3'], list) else [],
            'pdp_url': product['pdp_url'],
            'description': product['description'],
            'meta_info': product['meta_info'],
            'feature_list': product['feature_list'] if isinstance(product['feature_list'], list) else []
        }
    
    def get_trending_products(self, category=None, top_k=20):
        data_subset = self.data.copy()
        
        # Filter by category if specified
        if category:
            data_subset = data_subset[data_subset['category'] == category]
        
        # Filter out zero prices
        data_subset = data_subset[data_subset['selling_price_inr'] > 0]
        
        # Sort by launch date (most recent first)
        data_subset = data_subset.sort_values('launch_on', ascending=False)
        
        # Get top K trending products
        trending = data_subset.head(top_k)
        
        results = []
        for _, product in trending.iterrows():
            result = {
                'product_id': product['product_id'],
                'product_name': product['product_name'],
                'brand': product['brand'],
                'category': product['category'],
                'selling_price_inr': product['selling_price_inr'],
                'image_url': product['feature_image_s3'],
                'launch_date': product['launch_on']
            }
            results.append(result)
        
        return results
