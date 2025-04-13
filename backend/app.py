import os
import time
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
from werkzeug.utils import secure_filename
import traceback
import logging
from datetime import datetime
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = './uploads'
RESULT_FOLDER = './results'
MODEL_PATH = './models/denoiser_final.h5'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB
MAX_DIMENSION = 5000  # Max width/height in pixels

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Update the Limiter initialization
limiter = Limiter(
    get_remote_address,  # Pass the key_func directly
    default_limits=["200 per day", "50 per hour"]
)

# Attach the limiter to the app
limiter.init_app(app)

# Global model variable
model = None

def load_model_safely():
    """Attempt to load the model with fallback options"""
    global model

    # Define or import the custom function
    def mae(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    # Register the custom function
    tf.keras.utils.get_custom_objects()["mae"] = mae

    model_files = [
        './models/denoiser_final.h5',
        './models/denoiser_best.h5',
        './models/denoiser_partial.h5'
    ]

    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                logger.info(f"Attempting to load model from {model_file}")
                model = tf.keras.models.load_model(model_file, custom_objects={"mae": mae})
                logger.info(f"Successfully loaded model from {model_file}")
                return True
            except Exception as e:
                logger.error(f"Error loading model from {model_file}: {e}")
                continue

    logger.warning("No model could be loaded. Will run without denoising functionality.")
    return False

# Try to load the model at startup
load_model_safely()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files(directory, max_age_hours=1):
    """Clean up files older than max_age_hours in the specified directory"""
    now = time.time()
    cutoff = now - (max_age_hours * 3600)
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff:
                os.remove(filepath)
                logger.info(f"Cleaned up old file: {filepath}")
        except Exception as e:
            logger.error(f"Error cleaning up file {filepath}: {e}")

def add_noise(image, noise_type='gaussian', noise_level=0.1):
    """Add noise to an image for testing"""
    try:
        noisy_image = image.copy()
        noise_level = float(noise_level)
        
        if noise_type == 'gaussian':
            noise = np.random.normal(loc=0.0, scale=noise_level, size=image.shape)
            noisy_image = np.clip(image + noise, 0.0, 1.0)
        
        elif noise_type == 'poisson':
            scaled = image * 255.0
            noise = np.random.poisson(scaled * noise_level) / (255.0 * noise_level)
            noisy_image = np.clip(noise, 0.0, 1.0)
        
        elif noise_type == 'salt_pepper':
            s_vs_p = 0.5
            amount = noise_level
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            noisy_image[tuple(coords)] = 1
            num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            noisy_image[tuple(coords)] = 0
        
        return noisy_image
    except Exception as e:
        logger.error(f"Error adding noise: {e}")
        return image

def simple_denoise(img):
    """Apply simple denoising as a fallback if model is unavailable"""
    try:
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        return denoised, None
    except Exception as e:
        return None, str(e)

def process_image_tiled(model, image, tile_size=512, padding=32):
    """Process large images in tiles"""
    h, w = image.shape[:2]
    output = np.zeros_like(image)
    
    for y in range(0, h, tile_size - 2 * padding):
        for x in range(0, w, tile_size - 2 * padding):
            # Extract tile with padding
            y_start = max(0, y - padding)
            y_end = min(h, y + tile_size + padding)
            x_start = max(0, x - padding)
            x_end = min(w, x + tile_size + padding)
            
            tile = image[y_start:y_end, x_start:x_end]
            
            # Process tile
            processed_tile = model.predict(np.expand_dims(tile, 0))[0]
            
            # Blend the processed tile into the output (excluding padding)
            out_y_start = y if y > 0 else 0
            out_y_end = min(y + tile_size, h)
            out_x_start = x if x > 0 else 0
            out_x_end = min(x + tile_size, w)
            
            tile_y_start = padding if y > 0 else 0
            tile_y_end = tile_y_start + (out_y_end - out_y_start)
            tile_x_start = padding if x > 0 else 0
            tile_x_end = tile_x_start + (out_x_end - out_x_start)
            
            output[out_y_start:out_y_end, out_x_start:out_x_end] = \
                processed_tile[tile_y_start:tile_y_end, tile_x_start:tile_x_end]
    
    return output

def process_image(image_path, noise_type=None, noise_level=None):
    """Process an image with the denoising model"""
    try:
        # Clean up old files before processing
        cleanup_old_files(UPLOAD_FOLDER)
        cleanup_old_files(RESULT_FOLDER)
        
        # Read and validate image
        img = cv2.imread(image_path)
        if img is None:
            return None, "Could not read the image"
        
        if max(img.shape) > MAX_DIMENSION:
            return None, f"Image dimensions too large (max {MAX_DIMENSION}px allowed)"
        
        # Convert to RGB and normalize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        
        # Add noise if specified
        if noise_type and noise_level:
            img_rgb = add_noise(img_rgb, noise_type, noise_level)
        
        # If no model, use simple denoising
        if model is None:
            denoised_img = (img_rgb * 255).astype(np.uint8)
            denoised_img, error = simple_denoise(denoised_img)
            if error:
                return None, error
            return cv2.cvtColor(denoised_img, cv2.COLOR_RGB2BGR), None
        
        # Pad image if needed
        h, w = img_rgb.shape[:2]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            img_padded = np.pad(img_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            img_padded = img_rgb
        
        # Process based on image size
        if max(img_padded.shape[:2]) > 1024:
            # Use tiled processing for large images
            denoised_padded = process_image_tiled(model, img_padded)
        else:
            # Process whole image at once
            denoised_padded = model.predict(np.expand_dims(img_padded, 0))[0]
        
        # Crop to original size and convert back
        denoised_img = np.clip(denoised_padded[:h, :w, :], 0, 1)
        denoised_img_bgr = (denoised_img * 255).astype(np.uint8)
        denoised_img_bgr = cv2.cvtColor(denoised_img_bgr, cv2.COLOR_RGB2BGR)
        
        return denoised_img_bgr, None
    
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        logger.error(traceback.format_exc())
        return None, str(e)

@app.route('/upload', methods=['POST'])
@limiter.limit("10 per minute")
def upload_file():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Create secure filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        try:
            # Get noise parameters
            noise_type = request.form.get('noise_type')
            noise_level = request.form.get('noise_level')
            
            # Process the image
            denoised_img, error = process_image(file_path, noise_type, noise_level)
            
            if error:
                os.remove(file_path)
                return jsonify({'error': error}), 400
            
            # Save the result
            result_filename = f"denoised_{filename}"
            result_path = os.path.join(RESULT_FOLDER, result_filename)
            cv2.imwrite(result_path, denoised_img)
            
            return jsonify({
                'message': 'Image processed successfully',
                'original_image_path': f"/uploads/{filename}",
                'denoised_image_path': f"/results/{result_filename}"
            })
            
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
            
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/status', methods=['GET'])
def status():
    models_exist = os.path.exists('./models')
    models_available = []
    
    if models_exist:
        models_available = [f for f in os.listdir('./models') if f.endswith('.h5')]
    
    return jsonify({
        'status': 'online',
        'model_loaded': model is not None,
        'models_directory_exists': models_exist,
        'available_models': models_available,
        'server_time': datetime.now().isoformat(),
        'memory_usage': os.getpid(),
    })

@app.route('/')
def home():
    return """
    <html>
        <head>
            <title>Image Denoising API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                .endpoint { background: #f5f5f5; padding: 10px; border-radius: 5px; margin: 10px 0; }
                code { background: #eee; padding: 2px 5px; }
            </style>
        </head>
        <body>
            <h1>Image Denoising API</h1>
            <p>This service provides image denoising capabilities via a REST API.</p>
            
            <div class="endpoint">
                <h2>API Endpoints</h2>
                <p><strong>POST /upload</strong> - Upload an image for denoising</p>
                <p><strong>GET /status</strong> - Check server status and model availability</p>
                <p><strong>GET /uploads/{filename}</strong> - Access uploaded images</p>
                <p><strong>GET /results/{filename}</strong> - Access processed images</p>
            </div>
            
            <div class="endpoint">
                <h2>Upload Requirements</h2>
                <ul>
                    <li>Max file size: 10MB</li>
                    <li>Allowed formats: JPEG, PNG</li>
                    <li>Max dimensions: 5000×5000 pixels</li>
                </ul>
            </div>
            
            <p>For more information, please refer to the API documentation.</p>
        </body>
    </html>
    """
