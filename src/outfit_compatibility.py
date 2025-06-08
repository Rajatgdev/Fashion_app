import numpy as np
from color_analyzer import ColorAnalyzer
from metadata_extractor import MetadataExtractor

class OutfitCompatibilityEngine:
    def __init__(self):
        self.color_analyzer = ColorAnalyzer()
        self.metadata_extractor = MetadataExtractor()
    
    def recommend_complete_outfit(self, uploaded_metadata, all_items, top_k=6):
        """Simple outfit completion based on uploaded image"""
        recommendations = []
        
        # Get predicted category from uploaded image
        predicted_category = uploaded_metadata.get('predicted_category', 'unknown')
        primary_color = uploaded_metadata.get('primary_color', 'unknown')
        
        # Determine target category (opposite of uploaded)
        if predicted_category == 'dress':
            target_category = 'jeans'
        elif predicted_category == 'jeans':
            target_category = 'dress'
        else:
            target_category = None
        
        # Filter items by target category
        if target_category:
            target_items = [item for item in all_items if item.get('category') == target_category]
        else:
            target_items = all_items
        
        # Score items based on simple compatibility
        for item in target_items:
            if item.get('selling_price_inr', 0) > 0:  
                
                # Simple compatibility score based on color
                item_color = self.color_analyzer.get_color_from_text(item.get('combined_text', ''))
                
                # Basic color compatibility
                if primary_color != 'unknown':
                    color_compatibility = self.color_analyzer.calculate_color_compatibility(primary_color, item_color)
                else:
                    color_compatibility = 0.7  
                
                if item_color in ['black', 'white', 'grey', 'navy']:
                    color_compatibility = max(color_compatibility, 0.8)
                
                recommendations.append({
                    'product_id': item['product_id'],
                    'product_name': item['product_name'],
                    'brand': item['brand'],
                    'category': item['category'],
                    'selling_price_inr': item['selling_price_inr'],
                    'image_url': item['feature_image_s3'],
                    'compatibility_score': color_compatibility,
                    'pdp_url': item['pdp_url']
                })
        
        # Sort by compatibility score and return top items
        recommendations.sort(key=lambda x: x['compatibility_score'], reverse=True)
        return recommendations[:top_k]
