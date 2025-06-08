from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import os

class YoloFashionDetector:
    def __init__(self):
        try:
            # Get the correct path to your model
            current_dir = Path(__file__).parent.parent  
            model_path = current_dir / "models" / "yolo_training" / "fashion_classifier" / "weights" / "best.pt"
            
            print(f"🔍 Looking for model")
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = YOLO(str(model_path))
            print(f"🔥 Loaded custom YOLOv8 model successfully!")

        except Exception as e:
            print(f"❌ Error loading YOLOv8 model: {e}")
            print("⚠️ Will use fallback detection method")
            self.model = None
    
    def detect_category(self, image_path):
        if not self.model:
            print("⚠️ YOLOv8 model not loaded, using fallback")
            return self._fallback_detection(image_path)
        
        try:
            print(f"🔍 Running YOLOv8 inference")
            
            # Run inference with your custom model
            results = self.model(image_path, verbose=False)
            
            if results and len(results[0].boxes) > 0:
                # Get the highest confidence detection
                boxes = results[0].boxes
                class_ids = boxes.cls.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                
                # Get the detection with highest confidence
                best_idx = np.argmax(confidences)
                class_id = int(class_ids[best_idx])
                confidence = float(confidences[best_idx])
                
                # Map class_id to category (based on your training: 0=dress, 1=jeans)
                category_map = {0: 'dress', 1: 'jeans'}
                category = category_map.get(class_id, 'unknown')
                
                print(f"🎯 YOLOv8 detected: {category} (confidence: {confidence:.3f})")                
                return category, confidence
            else:
                print("⚠️ No objects detected by YOLOv8, using fallback")
                return self._fallback_detection(image_path)
                
        except Exception as e:
            print(f"❌ YOLOv8 detection error: {e}")
            return self._fallback_detection(image_path)
    
    def _fallback_detection(self, image_path):
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return 'dress', 0.5
                
            height, width = image.shape[:2]
            aspect_ratio = height / width
            
            category = 'dress' if aspect_ratio > 1.3 else 'jeans'
            print(f"📐 Fallback detection: {category} (aspect ratio: {aspect_ratio:.2f})")
            return category, 0.7
            
        except Exception as e:
            print(f"❌ Fallback detection error: {e}")
            return 'dress', 0.5
