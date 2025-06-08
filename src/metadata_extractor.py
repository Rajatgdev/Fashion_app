import cv2
import numpy as np
from pathlib import Path
from yolo_detector import YoloFashionDetector

class MetadataExtractor:
    def __init__(self):
        self.yolo_detector = YoloFashionDetector()
        
        # Print model status
        if self.yolo_detector.model:
            print("✅ YOLOv8 Custom Fashion Classifier loaded successfully!")
        else:
            print("⚠️ YOLOv8 model not loaded - using fallback methods")
    
    def detect_category_simple(self, image_path):
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Could not load image: {image_path}")
                return 'dress'
                
            height, width = image.shape[:2]
            aspect_ratio = height / width
            
            category = 'dress' if aspect_ratio > 1.3 else 'jeans'
            print(f"📐 Simple detection: {category} (aspect ratio: {aspect_ratio:.2f})")
            return category
            
        except Exception as e:
            print(f"❌ Simple detection error: {e}")
            return 'dress'
    
    def detect_category(self, image_path):
        if self.yolo_detector.model:
            category, confidence = self.yolo_detector.detect_category(image_path)
            return category
        else:
            # Fallback to simple method if YOLOv8 not available
            return self.detect_category_simple(image_path)
    
    def extract_dominant_colors(self, image_path, k=3):
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return []
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Take center region of image
            height, width = image_rgb.shape[:2]
            y_start = int(height * 0.3)
            y_end = int(height * 0.7)
            x_start = int(width * 0.3)
            x_end = int(width * 0.7)
            
            center_region = image_rgb[y_start:y_end, x_start:x_end]
            pixels = center_region.reshape((-1, 3))
            
            # Simple color analysis - get average colors
            colors = []
            if len(pixels) > 0:
                # Get dominant color (mean)
                mean_color = np.mean(pixels, axis=0).astype(int)
                colors.append((tuple(mean_color), 1.0))
            
            return colors
            
        except Exception as e:
            print(f"Error extracting colors: {e}")
            return []
    
    def get_closest_color_name(self, rgb_color):
        r, g, b = rgb_color
        
        # Simple color mapping
        if max(r, g, b) < 80:
            return 'black'
        elif min(r, g, b) > 200:
            return 'white'
        elif b > r + 30 and b > g + 30:
            return 'blue'
        elif r > g + 30 and r > b + 30:
            return 'red'
        elif g > r + 30 and g > b + 30:
            return 'green'
        elif abs(r - g) < 30 and abs(g - b) < 30:
            if max(r, g, b) > 150:
                return 'grey'
            else:
                return 'black'
        else:
            return 'unknown'
    
    def analyze_image_brightness(self, image_path):
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return 'medium'
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            
            if avg_brightness < 85:
                return 'dark'
            elif avg_brightness > 170:
                return 'light'
            else:
                return 'medium'
        except Exception as e:
            print(f"Error analyzing brightness: {e}")
            return 'medium'
    
    def extract_metadata(self, image_path):
        try:
            print(f"🔍 Extracting metadata")
            
            # Get category and confidence from YOLOv8
            if self.yolo_detector.model:
                category, confidence = self.yolo_detector.detect_category(image_path)
                detection_method = 'yolov8_custom'
                model_accuracy = 0.993
            else:
                category = self.detect_category_simple(image_path)
                confidence = 0.7
                detection_method = 'fallback_aspect_ratio'
                model_accuracy = 0.7
            
            # Get basic color info (optional)
            dominant_colors = self.extract_dominant_colors(image_path)
            primary_color = 'unknown'
            if dominant_colors:
                primary_color = self.get_closest_color_name(dominant_colors[0][0])
            
            # Get brightness
            brightness = self.analyze_image_brightness(image_path)
            
            # Create comprehensive metadata dict
            metadata = {
                'predicted_category': category,
                'detection_method': detection_method,
                'confidence': confidence,
                'model_accuracy': model_accuracy,
                'training_hours': 10.869 if detection_method == 'yolov8_custom' else 0,
                'primary_color': primary_color,
                'brightness': brightness,
                'dominant_colors': dominant_colors,
                'color_diversity': len(dominant_colors)
            }
            
            print(f"✅ Extracted metadata: {metadata}")
            return metadata
            
        except Exception as e:
            print(f"❌ Error extracting metadata: {e}")
            import traceback
            traceback.print_exc()
            return {
                'predicted_category': 'dress',
                'detection_method': 'error_fallback',
                'confidence': 0.5,
                'model_accuracy': 0.5,
                'primary_color': 'unknown',
                'brightness': 'medium',
                'dominant_colors': [],
                'color_diversity': 0
            }

# For backward compatibility with your existing code
def extract_metadata_from_image(image_path):
    extractor = MetadataExtractor()
    return extractor.extract_metadata(image_path)

# Additional utility functions for compatibility
def detect_category_from_image(image_path):
    extractor = MetadataExtractor()
    return extractor.detect_category(image_path)
