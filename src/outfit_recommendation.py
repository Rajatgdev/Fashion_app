import numpy as np
import pandas as pd
from collections import defaultdict
from visual_search import VisualSearchEngine
from color_analyzer import ColorAnalyzer
from config import *

class OutfitRecommendationEngine:
    def __init__(self):
        self.search_engine = VisualSearchEngine()
        self.color_analyzer = ColorAnalyzer()
    
    def recommend_complete_outfit(self, anchor_product_id, top_k=6):
        """Simple outfit completion: jeans → dresses, dresses → jeans"""
        # Get anchor product details
        anchor_product = self.search_engine.get_product_by_id(anchor_product_id)
        
        if not anchor_product:
            return []
        
        anchor_category = anchor_product['category']
        
        if anchor_category == 'dress':
            target_category = 'jeans'
        elif anchor_category == 'jeans':
            target_category = 'dress'
        else:
            return []  
        
        
        target_items = self.search_engine.data[
            self.search_engine.data['category'] == target_category
        ]
        
        # Filter out zero prices
        target_items = target_items[target_items['selling_price_inr'] > 0]
        
        if len(target_items) == 0:
            return []
        
        # Simple scoring based on price range compatibility
        anchor_price = anchor_product.get('selling_price_inr', 0)
        
        recommendations = []
        
        for _, item in target_items.iterrows():
            item_price = item['selling_price_inr']
            
            
            if anchor_price > 0 and item_price > 0:
                price_ratio = min(anchor_price, item_price) / max(anchor_price, item_price)
                compatibility_score = price_ratio  # Simple price-based compatibility
            else:
                compatibility_score = 0.5  # Default score
            
            recommendations.append({
                'product_id': item['product_id'],
                'product_name': item['product_name'],
                'brand': item['brand'],
                'category': item['category'],
                'selling_price_inr': item['selling_price_inr'],
                'image_url': item['feature_image_s3'],
                'compatibility_score': compatibility_score,
                'pdp_url': item['pdp_url']
            })
        
        # Sort by compatibility score and return top results
        recommendations.sort(key=lambda x: x['compatibility_score'], reverse=True)
        
        return recommendations[:top_k]
    
    def get_seasonal_recommendations(self, season, category=None, top_k=20):
        """Simple seasonal recommendations based on available data"""
        data_subset = self.search_engine.data.copy()
        
        # Filter by category if specified
        if category:
            data_subset = data_subset[data_subset['category'] == category]
        
        # Filter out zero prices
        data_subset = data_subset[data_subset['selling_price_inr'] > 0]
        
        # Sort by launch date (most recent first) as proxy for seasonal
        data_subset = data_subset.sort_values('launch_on', ascending=False)
        
        # Get top K items
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
                'compatibility_score': 0.8,  # Default score
                'pdp_url': product['pdp_url']
            }
            results.append(result)
        
        return results
    
    def get_occasion_recommendations(self, occasion, category=None, top_k=20):
        """Simple occasion-based recommendations"""
        return self.get_seasonal_recommendations('current', category, top_k)

if __name__ == "__main__":
    outfit_engine = OutfitRecommendationEngine()
    
    sample_data = outfit_engine.search_engine.data.head(1)
    if len(sample_data) > 0:
        sample_product_id = sample_data.iloc[0]['product_id']
        
        outfit_recs = outfit_engine.recommend_complete_outfit(sample_product_id, top_k=3)
        print("Complete outfit recommendations:")
        for rec in outfit_recs:
            print(f"- {rec['product_name']} (Compatibility: {rec['compatibility_score']:.3f})")
