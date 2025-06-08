from flask import Flask, render_template, request, jsonify, redirect, session, url_for
import os
import sys
from pathlib import Path
import uuid
from werkzeug.utils import secure_filename
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from visual_search import VisualSearchEngine
from outfit_recommendation import OutfitRecommendationEngine
from outfit_compatibility import OutfitCompatibilityEngine

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-this-in-production'
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'static' / 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

try:
    search_engine = VisualSearchEngine()
    outfit_engine = OutfitRecommendationEngine()
    outfit_compatibility_engine = OutfitCompatibilityEngine()
    print("All engines initialized successfully!")
except Exception as e:
    print(f"Error initializing engines: {e}")
    search_engine = None
    outfit_engine = None
    outfit_compatibility_engine = None
    personalization_engine = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main visual search page"""
    trending_dresses = []
    trending_jeans = []
    
    if search_engine:
        try:
            trending_dresses = search_engine.get_trending_products(category='dress', top_k=6)
            trending_jeans = search_engine.get_trending_products(category='jeans', top_k=6)
        except Exception as e:
            print(f"Error getting trending products: {e}")
    
    return render_template('index.html', 
                         trending_dresses=trending_dresses,
                         trending_jeans=trending_jeans)

@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        
        # Get product details from search engine
        if search_engine:
            product = search_engine.get_product_by_id(product_id)
            if not product:
                return jsonify({'success': False, 'error': 'Product not found'}), 404
        else:
            return jsonify({'success': False, 'error': 'Search engine not available'}), 500
        
        # Initialize cart if it doesn't exist
        if 'cart' not in session:
            session['cart'] = []
        
        # Create product item dictionary
        cart_item = {
            'product_id': product['product_id'],
            'product_name': product['product_name'],
            'image_url': product['image_url'],
            'category': product['category'],
            'selling_price_inr': product['selling_price_inr'],
            'brand': product['brand']
        }
        
        # Check if item already in cart
        existing_item = next((item for item in session['cart'] if item['product_id'] == product_id), None)
        
        if not existing_item:
            session['cart'].append(cart_item)
            session.modified = True
            message = 'Item added to cart!'
        else:
            message = 'Item already in cart!'
        
        return jsonify({
            'success': True, 
            'cart_count': len(session['cart']),
            'message': message
        })
        
    except Exception as e:
        print(f"Error adding to cart: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/remove_from_cart', methods=['POST'])
def remove_from_cart():
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        clear_all = data.get('clear_all', False)
        
        if 'cart' in session:
            if clear_all:
                # Clear entire cart
                session['cart'] = []
                message = 'Cart cleared successfully!'
            elif product_id:
                # Remove specific item
                session['cart'] = [item for item in session['cart'] if item['product_id'] != product_id]
                message = 'Item removed from cart!'
            else:
                return jsonify({'success': False, 'error': 'No product_id or clear_all specified'}), 400
            
            session.modified = True
        
        return jsonify({
            'success': True,
            'cart_count': len(session.get('cart', [])),
            'message': message
        })
        
    except Exception as e:
        print(f"Error removing from cart: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/cart')
def view_cart():
    cart_items = session.get('cart', [])
    return render_template('cart.html', cart_items=cart_items)

@app.route('/api/cart_count')
def get_cart_count():
    cart_count = len(session.get('cart', []))
    return jsonify({'cart_count': cart_count})

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'GET':
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(url_for('index'))
        
        file = request.files['image']
        if file.filename == '':
            return redirect(url_for('index'))
        
        if file and allowed_file(file.filename):
            try:
                # Save uploaded file
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                file_path = app.config['UPLOAD_FOLDER'] / unique_filename
                file.save(file_path)
                
                # Resize image if too large
                with Image.open(file_path) as img:
                    if img.size[0] > 1024 or img.size[1] > 1024:
                        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                        img.save(file_path)
                
                similar_products = []
                outfit_recommendations = []
                metadata = {}
                
                if search_engine:
                    print(f"🚀 Starting YOLOv8-enhanced visual search for: {file_path}")
                    
                    # Get similar products with YOLOv8 metadata
                    similar_products, metadata = search_engine.search_by_image_with_metadata(file_path, top_k=12)
                    
                    # YOLOv8 category detection 
                    predicted_category = metadata.get('predicted_category', 'unknown')
                    confidence = metadata.get('confidence', 0.0)
                    detection_method = metadata.get('detection_method', 'unknown')
                    model_accuracy = metadata.get('model_accuracy', 0.0)
                    training_hours = metadata.get('training_hours', 0.0)
                    
                    print(f"🎯 YOLOv8 detected: {predicted_category} (confidence: {confidence:.3f})")
                    print(f"🔥 Detection method: {detection_method}")
                    print(f"📊 Model accuracy: {model_accuracy*100:.1f}%")
                    if training_hours > 0:
                        print(f"⏱️ Training time: {training_hours:.3f} hours")
                    
                    # Use YOLOv8 results for outfit recommendations
                    if predicted_category == 'jeans':
                        target_category = 'dress'
                        print(f"👖 Jeans detected with {model_accuracy*100:.1f}% accuracy - recommending DRESSES")
                    elif predicted_category == 'dress':
                        target_category = 'jeans'
                        print(f"👗 Dress detected with {model_accuracy*100:.1f}% accuracy - recommending JEANS")
                    else:
                        target_category = 'dress'  # Default
                        print(f"❓ Unknown category - defaulting to DRESSES")
                    
                    # Get items from opposite category
                    all_data = search_engine.data
                    target_items = all_data[all_data['category'] == target_category]
                    target_items = target_items[target_items['selling_price_inr'] > 0]
                    print(f"Found {len(target_items)} valid {target_category} items")
                    
                    # Create outfit recommendations from opposite category
                    outfit_recommendations = []
                    for _, item in target_items.head(8).iterrows():
                        outfit_recommendations.append({
                            'product_id': item['product_id'],
                            'product_name': item['product_name'],
                            'brand': item['brand'],
                            'category': item['category'],
                            'selling_price_inr': item['selling_price_inr'],
                            'image_url': item['feature_image_s3'],
                            'compatibility_score': 0.85,
                            'pdp_url': item['pdp_url']
                        })
                    
                    print(f"Created {len(outfit_recommendations)} outfit recommendations")
                    if outfit_recommendations:
                        print(f"Outfit recommendations are: {outfit_recommendations[0]['category']}")
                
                return render_template('outfit_results.html', 
                                     similar_products=similar_products,
                                     outfit_recommendations=outfit_recommendations,
                                     metadata=metadata,
                                     uploaded_image=f"uploads/{unique_filename}")
                                     
            except Exception as e:
                print(f"Error in YOLOv8-enhanced search: {e}")
                import traceback
                traceback.print_exc()
                return redirect(url_for('index'))
        
        return redirect(url_for('index'))

@app.route('/product/<product_id>')
def product_detail(product_id):
    if not search_engine:
        return "Search engine not available", 500
    
    try:
        product = search_engine.get_product_by_id(product_id)
        
        if not product:
            return "Product not found", 404
        
        # Get outfit recommendations
        outfit_recommendations = []
        similar_products = []
        
        if outfit_engine:
            try:
                outfit_recommendations = outfit_engine.recommend_complete_outfit(product_id, top_k=6)
                outfit_recommendations = [r for r in outfit_recommendations if r.get('selling_price_inr', 0) > 0]
            except Exception as e:
                print(f"Error getting outfit recommendations: {e}")
        
        # Get visually similar products using the product's own image
        try:
            product_image_path = f"data/images/{product_id}.jpg"
            
            if Path(product_image_path).exists():
                similar_results = search_engine.search_by_image(product_image_path, top_k=10)
                
                # Remove the current product from results
                for result in similar_results:
                    if (result['product_id'] != product_id and 
                        result.get('selling_price_inr', 0) > 0 and 
                        len(similar_products) < 6):
                        similar_products.append(result)
                        
        except Exception as e:
            print(f"Error getting visually similar products: {e}")
        
        return render_template('product_detail.html', 
                             product=product,
                             outfit_recommendations=outfit_recommendations,
                             similar_products=similar_products)
                             
    except Exception as e:
        print(f"Error in product detail: {e}")
        return f"Error loading product: {str(e)}", 500

@app.route('/api/visual_search', methods=['POST'])
def api_visual_search():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file and allowed_file(file.filename):
        # Save and process image
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = app.config['UPLOAD_FOLDER'] / unique_filename
        file.save(file_path)
        
        
        if search_engine:
            try:
                similar_products, metadata = search_engine.search_by_image_with_metadata(file_path, top_k=20)
                
                # Get outfit recommendations based on YOLOv8 detection
                predicted_category = metadata.get('predicted_category', 'unknown')
                
                # Get items from opposite category for outfit recommendations
                all_data = search_engine.data
                if predicted_category == 'jeans':
                    target_category = 'dress'
                elif predicted_category == 'dress':
                    target_category = 'jeans'
                else:
                    target_category = 'dress'
                
                target_items = all_data[all_data['category'] == target_category]
                target_items = target_items[target_items['selling_price_inr'] > 0]
                
                outfit_recommendations = []
                for _, item in target_items.head(10).iterrows():
                    outfit_recommendations.append({
                        'product_id': item['product_id'],
                        'product_name': item['product_name'],
                        'brand': item['brand'],
                        'category': item['category'],
                        'selling_price_inr': item['selling_price_inr'],
                        'image_url': item['feature_image_s3'],
                        'compatibility_score': 0.85,
                        'pdp_url': item['pdp_url']
                    })
                
                return jsonify({
                    'similar_products': similar_products,
                    'outfit_recommendations': outfit_recommendations,
                    'metadata': metadata,
                    'uploaded_image': f"uploads/{unique_filename}",
                    'yolo_detection': {
                        'category': predicted_category,
                        'confidence': metadata.get('confidence', 0.0),
                        'model_accuracy': metadata.get('model_accuracy', 0.0)
                    }
                })
                
            except Exception as e:
                print(f"Error in API visual search: {e}")
                return jsonify({'error': f'Search processing failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid image file'}), 400

@app.route('/api/model_info')
def get_model_info():
    try:
        
        from metadata_extractor import MetadataExtractor
        
        extractor = MetadataExtractor()
        if hasattr(extractor, 'yolo_detector') and extractor.yolo_detector.model:
            model_info = {
                'model_loaded': True,
                'model_type': 'YOLOv8n Custom Fashion Classifier',
                'accuracy': '99.3% mAP@50',
                'training_time': '10.869 hours',
                'classes': ['dress', 'jeans'],
                'detection_method': 'yolov8_custom'
            }
        else:
            model_info = {
                'model_loaded': False,
                'fallback_method': 'Aspect ratio detection',
                'detection_method': 'fallback'
            }
        
        return jsonify(model_info)
        
    except Exception as e:
        return jsonify({
            'model_loaded': False,
            'error': str(e),
            'detection_method': 'error'
        }), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)
