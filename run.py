"""
Enhanced Fashion Visual Search & Intelligent Styling Assistant
Main execution script for the complete pipeline
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

def run_preprocessing():
    print("Starting data preprocessing...")
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    data = preprocessor.run_preprocessing(max_images=7000)  # Process more images
    print(f"Preprocessing completed! Processed {len(data)} products.")

def run_feature_extraction():
    print("Starting feature extraction...")
    from feature_extraction import FeatureExtractor, FAISSIndexer
    import pickle
    from config import PROCESSED_DATA_FILE, PROCESSED_DATA_DIR
    
    # Load processed data
    with open(PROCESSED_DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    
    # Extract features
    extractor = FeatureExtractor()
    image_features, text_features, valid_indices = extractor.extract_all_features(data)
    
    # Create combined features with higher weight on visual features
    combined_features = extractor.create_combined_features(image_features, text_features, alpha=0.9)
    
    # Save features
    extractor.save_features(image_features, text_features, combined_features, valid_indices)
    
    # Build FAISS index
    indexer = FAISSIndexer()
    product_ids = [data.iloc[i]['product_id'] for i in valid_indices]
    indexer.build_index(combined_features, product_ids)
    indexer.save_index(PROCESSED_DATA_DIR / 'faiss_index.bin')
    
    print("Feature extraction and indexing completed!")

def run_web_app():
    """Run the enhanced web application"""
    print("Starting enhanced web application...")
    from app.main import app
    app.run(debug=True, host='0.0.0.0', port=5001)

def run_tests():
    """Run system tests"""
    print("Running enhanced system tests...")
    from visual_search import VisualSearchEngine
    from outfit_compatibility import OutfitCompatibilityEngine
    
    # Test enhanced visual search
    search_engine = VisualSearchEngine()
    
    # Test with a sample image
    sample_images = list(Path("data/images").glob("*.jpg"))
    if sample_images:
        test_image = sample_images[0]
        print(f"Testing with image: {test_image}")
        
        similar_products, metadata = search_engine.search_by_image_with_metadata(test_image, top_k=5)
        print(f"Enhanced search test: Found {len(similar_products)} similar products")
        print(f"Extracted metadata: {metadata}")
        
        # Test outfit recommendations
        outfit_engine = OutfitCompatibilityEngine()
        if metadata:
            all_items = search_engine.data.to_dict('records')
            outfit_recs = outfit_engine.recommend_complete_outfit(metadata, all_items, top_k=3)
            print(f"Outfit recommendation test: Found {len(outfit_recs)} outfit suggestions")
    
    print("All enhanced tests completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Fashion Visual Search System')
    parser.add_argument('command', choices=['preprocess', 'extract', 'web', 'test', 'all'], 
                       help='Command to run')
    
    args = parser.parse_args()
    
    if args.command == 'preprocess':
        run_preprocessing()
    elif args.command == 'extract':
        run_feature_extraction()
    elif args.command == 'web':
        run_web_app()
    elif args.command == 'test':
        run_tests()
    elif args.command == 'all':
        print("Running complete enhanced pipeline...")
        run_preprocessing()
        run_feature_extraction()
        run_tests()
        print("Enhanced pipeline completed! Run 'python run.py web' to start the web app.")

if __name__ == '__main__':
    main()
