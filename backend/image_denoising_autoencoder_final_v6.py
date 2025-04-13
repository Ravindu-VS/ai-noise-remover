import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# Paths
CLEAN_DATA_PATH = os.path.abspath("../train_data/clean/dummy")  
OUTPUT_PATH = "./output"
MODEL_PATH = "./models"

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Configuration
CONFIG = {
    'batch_size': 8,  # Reduced for CPU processing
    'patch_size': 96,  # Smaller patches for CPU
    'epochs': 200,     # Further increased for more training


    'initial_lr': 0.00001,  # Further reduced for finer adjustments


    'min_lr': 0.00001,
    'noise_levels': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],  # Added even higher noise level for robustness


    'noise_types': ['gaussian', 'poisson', 'salt_pepper'],
    'model_type': 'dncnn',
    'loss_type': 'mae',  # Changed to MAE for stability
}

print(f"TensorFlow version: {tf.__version__}")
print("Available devices:", tf.config.list_physical_devices())

# Custom noise generator with multiple noise types
def add_noise(image, noise_type='gaussian', noise_level=0.1):
    noisy_image = image.copy()
    
    if noise_type == 'gaussian':
        noise = np.random.normal(loc=0.0, scale=noise_level, size=image.shape)
        noisy_image = np.clip(image + noise, 0.0, 1.0)
    
    elif noise_type == 'poisson':
        # Scale to avoid very low intensity issues
        scaled = image * 255.0
        noise = np.random.poisson(scaled * noise_level) / (255.0 * noise_level)
        noisy_image = np.clip(noise, 0.0, 1.0)
    
    elif noise_type == 'salt_pepper':
        s_vs_p = 0.5
        amount = noise_level
        # Salt
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[tuple(coords)] = 1
        
        # Pepper
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[tuple(coords)] = 0
    
    return noisy_image

# Custom data generator with patches and noise
class NoisyPatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, filepaths, batch_size, patch_size, noise_types, noise_levels):
        self.filepaths = filepaths
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.noise_types = noise_types
        self.noise_levels = noise_levels
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.filepaths) * 4 / self.batch_size))  # 4 patches per image
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.filepaths))
        np.random.shuffle(self.indices)
    
    def extract_random_patch(self, image):
        h, w = image.shape[:2]
        
        if h > self.patch_size and w > self.patch_size:
            top = np.random.randint(0, h - self.patch_size)
            left = np.random.randint(0, w - self.patch_size)
            return image[top:top+self.patch_size, left:left+self.patch_size]
        else:
            # Resize if image is smaller than patch size
            return cv2.resize(image, (self.patch_size, self.patch_size))
    
    def augment_patch(self, patch):
        # Random rotations and flips
        k = np.random.randint(0, 4)  # 0-3 for 0, 90, 180, 270 degrees
        if k > 0:
            patch = np.rot90(patch, k)
        
        # Random flip
        if np.random.rand() > 0.5:
            patch = np.fliplr(patch)
        
        return patch
    
    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        
        # Get a batch of indices
        batch_indices = []
        for i in range(self.batch_size):
            index = self.indices[(idx * self.batch_size + i) % len(self.indices)]
            batch_indices.append(index)
        
        for index in batch_indices:
            img_path = self.filepaths[index]
            
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                
                # Extract patch
                patch = self.extract_random_patch(img)
                patch = self.augment_patch(patch)
                
                # Select random noise type and level
                noise_type = np.random.choice(self.noise_types)
                noise_level = np.random.choice(self.noise_levels)
                
                # Add noise
                noisy_patch = add_noise(patch, noise_type, noise_level)
                
                batch_x.append(noisy_patch)
                batch_y.append(patch)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Ensure we have something in the batch
        if not batch_x:
            # Generate a default patch if all images failed
            default_patch = np.zeros((self.patch_size, self.patch_size, 3))
            batch_x = [default_patch] * self.batch_size
            batch_y = [default_patch] * self.batch_size
        
        # Pad batch if necessary to match batch_size
        while len(batch_x) < self.batch_size:
            batch_x.append(batch_x[0])
            batch_y.append(batch_y[0])
            
        return np.array(batch_x), np.array(batch_y)

# Build DnCNN model (optimized for denoising)
def build_dncnn_model(input_shape=(None, None, 3), depth=10):  # Reduced depth for CPU
    inputs = layers.Input(shape=input_shape)
    
    # First layer
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.Activation('relu')(x)  # Using standard ReLU for compatibility
    
    # Middle layers
    for _ in range(depth-2):
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    
    # Last layer (outputs the noise)
    x = layers.Conv2D(3, (3, 3), padding='same')(x)
    
    # Residual learning (subtract noise instead of predicting clean image)
    outputs = layers.Subtract()([inputs, x])
    
    model = models.Model(inputs, outputs)
    return model

# Build U-Net model (better at preserving details)
def build_unet_model(input_shape=(None, None, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)  # Reduced filters
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Bridge
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
    
    # Decoder with skip connections
    up4 = layers.UpSampling2D(size=(2, 2))(conv3)
    up4 = layers.Conv2D(64, 2, activation='relu', padding='same')(up4)
    merge4 = layers.Concatenate()([conv2, up4])
    conv4 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge4)
    conv4 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv4)
    
    up5 = layers.UpSampling2D(size=(2, 2))(conv4)
    up5 = layers.Conv2D(32, 2, activation='relu', padding='same')(up5)
    merge5 = layers.Concatenate()([conv1, up5])
    conv5 = layers.Conv2D(32, 3, activation='relu', padding='same')(merge5)
    conv5 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv5)
    
    outputs = layers.Conv2D(3, 1, activation='sigmoid')(conv5)
    
    model = models.Model(inputs, outputs)
    return model

# Training function
def train_denoising_model():
    # Get list of clean images
    clean_files = []
    for root, _, files in os.walk(CLEAN_DATA_PATH):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                clean_files.append(os.path.join(root, file))
    
    # Ensure we have images
    if not clean_files:
        raise ValueError(f"No images found in {CLEAN_DATA_PATH}")
    
    print(f"Found {len(clean_files)} clean images")
    
    # Validate images
    valid_files = []
    for file_path in clean_files:
        try:
            img = cv2.imread(file_path)
            if img is not None:
                valid_files.append(file_path)
            else:
                print(f"Warning: Could not read {file_path}")
        except Exception as e:
            print(f"Error with {file_path}: {e}")
    
    print(f"Valid images: {len(valid_files)}/{len(clean_files)}")
    
    # Create data generator
    train_gen = NoisyPatchGenerator(
        valid_files, 
        CONFIG['batch_size'], 
        CONFIG['patch_size'], 
        CONFIG['noise_types'], 
        CONFIG['noise_levels']
    )
    
    # Build the model
    if CONFIG['model_type'] == 'dncnn':
        model = build_dncnn_model(input_shape=(CONFIG['patch_size'], CONFIG['patch_size'], 3))
        print("Using DnCNN model")
    else:
        model = build_unet_model(input_shape=(CONFIG['patch_size'], CONFIG['patch_size'], 3))
        print("Using U-Net model")
    
    model.summary()
    
    # Set up optimizer with learning rate
    optimizer = optimizers.Adam(learning_rate=CONFIG['initial_lr'])
    
    # Configure loss
    if CONFIG['loss_type'] == 'mse':
        loss = 'mse'
        print("Using MSE loss")
    elif CONFIG['loss_type'] == 'mae':
        loss = 'mae'
        print("Using MAE loss")
    else:
        loss = 'mae'  # Default to MAE
        print("Using MAE loss (default)")
    
    # Compile
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_PATH, 'denoiser_best.h5'),
            monitor='loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=3,
            min_lr=CONFIG['min_lr'],
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(OUTPUT_PATH, 'logs'),
            update_freq='epoch'
        )
    ]
    
    # Train
    try:
        history = model.fit(
            train_gen,
            epochs=CONFIG['epochs'],
            callbacks=callbacks
            # Removed workers and multiprocessing arguments
        )
        
        # Save the final model
        model.save(os.path.join(MODEL_PATH, 'denoiser_final.h5'))
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Loss')
        plt.plot(history.history['mae'], label='MAE')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss/MAE')
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_PATH, 'training_history.png'))
        
        return model
    
    except Exception as e:
        print(f"Training error: {e}")
        # Try to save the model anyway
        try:
            model.save(os.path.join(MODEL_PATH, 'denoiser_partial.h5'))
            print("Saved partial model")
        except:
            pass
        raise e

# Test function
def test_model(model, test_image_path=None, noise_level=0.1, noise_type='gaussian'):
    # Use a random image from training set
    clean_files = [f for f in os.listdir(CLEAN_DATA_PATH) 
                 if os.path.isfile(os.path.join(CLEAN_DATA_PATH, f)) and 
                 f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not clean_files:
        print("No test images found!")
        return None, None, None
    
    img_path = np.random.choice(clean_files)
    img = cv2.imread(os.path.join(CLEAN_DATA_PATH, img_path))
    
    if img is None:
        print(f"Could not load test image {img_path}")
        return None, None, None
            
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    
    # Resize for processing
    img = cv2.resize(img, (96, 96))  # Resize to 96x96
    
    # Create noisy version
    noisy_img = add_noise(img, noise_type, noise_level)
    
    # Ensure dimensions are divisible by 8 for the network
    h, w = img.shape[:2]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    
    if pad_h > 0 or pad_w > 0:
        noisy_padded = np.pad(noisy_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        noisy_padded = noisy_img
    
    # Denoise
    try:
        denoised_padded = model.predict(np.expand_dims(noisy_padded, 0))[0]
        # Crop back to original size
        denoised_img = denoised_padded[:h, :w, :]
        denoised_img = np.clip(denoised_img, 0, 1)
    except Exception as e:
        print(f"Prediction error: {e}")
        return img, noisy_img, None
    
    # Display results
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title(f'Noisy ({noise_type}, level={noise_level})')
    plt.imshow(noisy_img)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Denoised')
    plt.imshow(denoised_img)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, f'denoising_result_{noise_type}_{noise_level}.png'))
    plt.show()
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    def calculate_psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    # Calculate metrics for noisy image
    noisy_psnr = calculate_psnr(img, noisy_img)
    
    # Calculate metrics for denoised image
    denoised_psnr = calculate_psnr(img, denoised_img)
    
    print(f"Noisy Image - PSNR: {noisy_psnr:.2f} dB")
    print(f"Denoised Image - PSNR: {denoised_psnr:.2f} dB")
    print(f"Improvement - PSNR: {denoised_psnr-noisy_psnr:.2f} dB")
    
    return img, noisy_img, denoised_img

# Main execution
if __name__ == "__main__":
    print(f"Starting advanced denoising training with configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    try:
        # Train model
        model = train_denoising_model()
        
        # Test on different noise types
        for noise_type in CONFIG['noise_types']:
            for noise_level in [0.05, 0.15]:
                print(f"\nTesting on {noise_type} noise with level {noise_level}")
                test_model(model, noise_level=noise_level, noise_type=noise_type)
    
    except Exception as e:
        print(f"Error in main execution: {e}")
