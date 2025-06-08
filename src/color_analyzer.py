import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ColorAnalyzer:
    def __init__(self):
        self.color_harmony_rules = {
            'complementary': self._get_complementary_colors,
            'analogous': self._get_analogous_colors,
            'triadic': self._get_triadic_colors,
            'neutral': self._get_neutral_colors
        }
        
        self.color_categories = {
            'warm': ['red', 'orange', 'yellow', 'pink'],
            'cool': ['blue', 'green', 'purple', 'navy'],
            'neutral': ['black', 'white', 'grey', 'beige', 'brown'],
            'bright': ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink'],
            'dark': ['black', 'navy', 'brown'],
            'light': ['white', 'beige', 'pink']
        }
    
    def _get_complementary_colors(self, color):
        """Get complementary colors"""
        complementary_map = {
            'red': ['green', 'blue'],
            'blue': ['orange', 'yellow'],
            'green': ['red', 'pink'],
            'yellow': ['purple', 'blue'],
            'orange': ['blue', 'navy'],
            'purple': ['yellow', 'green'],
            'pink': ['green', 'navy'],
            'black': ['white', 'beige'],
            'white': ['black', 'navy'],
            'grey': ['yellow', 'orange'],
            'brown': ['blue', 'green'],
            'navy': ['orange', 'pink'],
            'beige': ['brown', 'navy']
        }
        return complementary_map.get(color, [])
    
    def _get_analogous_colors(self, color):
        """Get analogous colors"""
        analogous_map = {
            'red': ['orange', 'pink'],
            'blue': ['green', 'purple'],
            'green': ['blue', 'yellow'],
            'yellow': ['orange', 'green'],
            'orange': ['red', 'yellow'],
            'purple': ['blue', 'pink'],
            'pink': ['red', 'purple'],
            'black': ['grey', 'brown'],
            'white': ['beige', 'grey'],
            'grey': ['black', 'white'],
            'brown': ['beige', 'orange'],
            'navy': ['blue', 'purple'],
            'beige': ['brown', 'white']
        }
        return analogous_map.get(color, [])
    
    def _get_triadic_colors(self, color):
        """Get triadic colors"""
        triadic_map = {
            'red': ['blue', 'yellow'],
            'blue': ['red', 'yellow'],
            'yellow': ['red', 'blue'],
            'green': ['orange', 'purple'],
            'orange': ['green', 'purple'],
            'purple': ['green', 'orange']
        }
        return triadic_map.get(color, [])
    
    def _get_neutral_colors(self, color):
        """Get neutral colors that go with any color"""
        return ['black', 'white', 'grey', 'beige', 'navy']
    
    def calculate_color_compatibility(self, color1, color2):
        """Calculate compatibility score between two colors"""
        if color1 == color2:
            return 1.0
        
        # Check if colors are in neutral category
        if color1 in self.color_categories['neutral'] or color2 in self.color_categories['neutral']:
            return 0.9
        
        # Check harmony rules
        compatibility_score = 0.0
        
        for rule_name, rule_func in self.color_harmony_rules.items():
            compatible_colors = rule_func(color1)
            if color2 in compatible_colors:
                if rule_name == 'complementary':
                    compatibility_score = max(compatibility_score, 0.8)
                elif rule_name == 'analogous':
                    compatibility_score = max(compatibility_score, 0.7)
                elif rule_name == 'triadic':
                    compatibility_score = max(compatibility_score, 0.6)
                elif rule_name == 'neutral':
                    compatibility_score = max(compatibility_score, 0.9)
        
        # Check if colors are in same category
        for category, colors in self.color_categories.items():
            if color1 in colors and color2 in colors:
                compatibility_score = max(compatibility_score, 0.6)
        
        return compatibility_score
    
    def get_color_from_text(self, text):
        """Extract color from product text"""
        text_lower = text.lower()
        
        color_keywords = {
            'black': ['black', 'noir'],
            'white': ['white', 'cream', 'ivory', 'off-white'],
            'red': ['red', 'crimson', 'burgundy', 'wine', 'maroon'],
            'blue': ['blue', 'navy', 'denim', 'cobalt', 'royal blue'],
            'green': ['green', 'olive', 'emerald', 'forest'],
            'yellow': ['yellow', 'gold', 'mustard', 'lemon'],
            'orange': ['orange', 'coral', 'peach', 'tangerine'],
            'purple': ['purple', 'violet', 'lavender', 'plum'],
            'pink': ['pink', 'rose', 'blush', 'magenta'],
            'brown': ['brown', 'tan', 'camel', 'chocolate', 'coffee'],
            'grey': ['grey', 'gray', 'silver', 'charcoal'],
            'beige': ['beige', 'nude', 'sand', 'khaki']
        }
        
        for color, keywords in color_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return color
        
        return 'unknown'
