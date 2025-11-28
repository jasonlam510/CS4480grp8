#!/usr/bin/env python3
"""
Train CNN Model for Northern Lights Detection

Trains a binary classification CNN model to detect northern lights in images.
Loads data from CSV file, maps labels to binary classification, and trains with GPU/CUDA support.

Usage:
    python3 train_northern_lights_cnn.py --csv data/frames_x1_local_paths.csv

Example:
    python3 train_northern_lights_cnn.py --csv /home/jason/Github/CS4480grp8/data/frames_x10_predictions.csv --batch-size 32 --epochs 50
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


def get_gpu_info():
    """
    Get GPU information including CUDA cores, memory, and compute capability.
    
    Returns:
        dict: GPU information dictionary
    """
    gpu_info = {
        'gpu_available': False,
        'gpu_name': None,
        'cuda_cores': None,
        'total_memory_gb': None,
        'compute_capability': None,
        'cuda_version': None
    }
    
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        gpu_info['gpu_available'] = True
        gpu = gpus[0]
        
        # Get GPU details from TensorFlow
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            if gpu_details:
                gpu_info['gpu_name'] = gpu_details.get('device_name', 'Unknown')
                compute_cap = gpu_details.get('compute_capability', None)
                if compute_cap:
                    # Format as string if it's a tuple/list
                    if isinstance(compute_cap, (list, tuple)):
                        gpu_info['compute_capability'] = f"{compute_cap[0]}.{compute_cap[1]}"
                    else:
                        gpu_info['compute_capability'] = str(compute_cap)
        except Exception as e:
            pass
        
        # Try to get CUDA version
        try:
            build_info = tf.sysconfig.get_build_info()
            if 'cuda_version' in build_info:
                gpu_info['cuda_version'] = build_info['cuda_version']
        except Exception:
            pass
        
        # Get memory info - use 'limit' for total memory
        try:
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            if 'limit' in memory_info:
                gpu_info['total_memory_gb'] = memory_info['limit'] / (1024**3)
        except Exception:
            pass
        
        # Try to get detailed GPU info from nvidia-smi (if available)
        try:
            import subprocess
            # Get GPU name
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_info['gpu_name'] = result.stdout.strip()
            
            # Get total memory from nvidia-smi (more reliable)
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                try:
                    memory_mb = int(result.stdout.strip().split()[0])
                    gpu_info['total_memory_gb'] = memory_mb / 1024.0
                except (ValueError, IndexError):
                    pass
            
            # Try to get CUDA cores - nvidia-smi doesn't directly report this,
            # but we can try to get GPU architecture info
            # Note: CUDA cores need to be looked up from GPU model name
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], 
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and not gpu_info.get('compute_capability'):
                gpu_info['compute_capability'] = result.stdout.strip()
                
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            # nvidia-smi not available or failed
            pass
    
    return gpu_info


def map_label_to_binary(label):
    """
    Map original label to binary classification.
    
    Args:
        label: Original label string
        
    Returns:
        int: 1 for northern lights, 0 for others
    """
    label_lower = label.lower().strip()
    
    # Northern lights labels map to 1 (true)
    if 'northern light' in label_lower:
        return 1
    
    # All other labels map to 0 (false)
    return 0


def load_data_from_csv(csv_path):
    """
    Load dataset from CSV file.
    
    Args:
        csv_path: Path to CSV file with local_path and label columns
        
    Returns:
        tuple: (paths, labels) as lists
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    paths = []
    labels = []
    original_labels = []
    
    print(f"Loading data from CSV: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Validate required columns
        if 'local_path' not in reader.fieldnames or 'label' not in reader.fieldnames:
            raise ValueError("CSV must contain 'local_path' and 'label' columns")
        
        for row in reader:
            local_path = row['local_path'].strip()
            label = row['label'].strip()
            
            # Skip empty rows
            if not local_path or not label:
                continue
            
            # Handle quoted paths
            local_path = local_path.strip('"')
            
            # Convert to absolute path if relative
            if not os.path.isabs(local_path):
                local_path = os.path.abspath(local_path)
            
            paths.append(local_path)
            original_labels.append(label)
            labels.append(map_label_to_binary(label))
    
    print(f"Loaded {len(paths)} samples from CSV")
    return paths, labels, original_labels


def detect_image_size(paths):
    """
    Detect image size from first valid image.
    
    Args:
        paths: List of image paths
        
    Returns:
        tuple: (width, height) image size
    """
    print("Detecting image size from first image...")
    for path in paths:
        try:
            img = Image.open(path)
            image_size = img.size  # (width, height)
            print(f"Detected image size: {image_size[0]}x{image_size[1]}")
            return image_size
        except Exception as e:
            continue
    
    raise ValueError("Could not detect image size from any image.")


class DataGenerator(tf.keras.utils.Sequence):
    """
    Data generator that loads images in batches on-the-fly.
    This allows training on datasets larger than available RAM.
    """
    
    def __init__(self, paths, labels, image_size, batch_size=32, shuffle=True, augment=False, **kwargs):
        """
        Initialize data generator.
        
        Args:
            paths: List of image paths
            labels: List of binary labels
            image_size: Target image size (width, height)
            batch_size: Batch size
            shuffle: Whether to shuffle data after each epoch
            augment: Whether to apply data augmentation
            **kwargs: Additional arguments for Sequence base class (workers, use_multiprocessing, etc.)
        """
        super().__init__(**kwargs)
        self.paths = paths
        self.labels = np.array(labels, dtype=np.int32)
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.paths))
        
        # Create data augmentation layer if needed
        if self.augment:
            self.augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip('horizontal'),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomBrightness(0.1),
            ])
        else:
            self.augmentation = None
        
        self.on_epoch_end()
    
    def __len__(self):
        """Return number of batches per epoch."""
        return int(np.ceil(len(self.paths) / self.batch_size))
    
    def __getitem__(self, idx):
        """Generate one batch of data."""
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Initialize batch arrays
        batch_images = []
        batch_labels = []
        
        # Load images for this batch
        for i in batch_indices:
            try:
                img = Image.open(self.paths[i])
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if needed (though frames should already be resized)
                if img.size != self.image_size:
                    img = img.resize(self.image_size, Image.Resampling.LANCZOS)
                
                # Convert to numpy array and normalize to [0, 1]
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                batch_images.append(img_array)
                batch_labels.append(self.labels[i])
                
            except Exception as e:
                # If image fails to load, skip it (or use a zero image)
                # For robustness, we'll skip failed images
                continue
        
        # Convert to numpy arrays
        if len(batch_images) == 0:
            # Return empty batch if all images failed to load
            return np.array([]), np.array([])
        
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        
        # Apply augmentation if enabled
        if self.augment and self.augmentation is not None and len(batch_images) > 0:
            # Convert to tensor for augmentation
            batch_images_tensor = tf.convert_to_tensor(batch_images)
            batch_images = self.augmentation(batch_images_tensor, training=True).numpy()
        
        return batch_images, batch_labels
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


def build_cnn_model(input_shape, num_classes=1):
    """
    Build CNN model for binary classification.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes (1 for binary)
        
    Returns:
        tf.keras.Model: Compiled model
    """
    model = tf.keras.Sequential([
        # First convolutional block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Second convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Third convolutional block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Fourth convolutional block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Flatten and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])
    
    return model


def train_model(model, train_generator, val_generator, num_train_samples, num_val_samples,
                batch_size=32, epochs=50, learning_rate=0.001, model_dir='models'):
    """
    Train the CNN model using data generators.
    
    Args:
        model: Keras model to train
        train_generator: Training data generator
        val_generator: Validation data generator
        num_train_samples: Number of training samples
        num_val_samples: Number of validation samples
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        model_dir: Directory to save model
        
    Returns:
        tuple: (trained_model, history, performance_log)
    """
    # Compile model
    # For binary classification, use explicit threshold for precision/recall
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision', thresholds=0.5),
            tf.keras.metrics.Recall(name='recall', thresholds=0.5)
        ]
    )
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'checkpoint.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Performance tracking
    performance_log = {
        'gpu_info': get_gpu_info(),
        'training_start': datetime.now().isoformat(),
        'epochs': [],
        'batch_times': []
    }
    
    print("\n" + "="*60)
    print("GPU Information:")
    print("="*60)
    for key, value in performance_log['gpu_info'].items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Train model
    print(f"Starting training...")
    print(f"  Training samples: {num_train_samples}")
    print(f"  Validation samples: {num_val_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Using data generators (memory-efficient)\n")
    
    start_time = datetime.now()
    
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks,
        verbose=1
    )
    
    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()
    
    # Calculate performance metrics
    total_samples = num_train_samples * epochs
    throughput = total_samples / training_duration if training_duration > 0 else 0
    
    performance_log['training_end'] = datetime.now().isoformat()
    performance_log['batch_size'] = batch_size
    performance_log['epochs'] = epochs
    performance_log['learning_rate'] = learning_rate
    performance_log['num_train_samples'] = num_train_samples
    performance_log['num_val_samples'] = num_val_samples
    performance_log['training_duration_seconds'] = training_duration
    performance_log['total_samples_processed'] = total_samples
    performance_log['throughput_images_per_second'] = throughput
    performance_log['final_train_accuracy'] = float(history.history['accuracy'][-1])
    performance_log['final_val_accuracy'] = float(history.history['val_accuracy'][-1])
    performance_log['final_train_loss'] = float(history.history['loss'][-1])
    performance_log['final_val_loss'] = float(history.history['val_loss'][-1])
    
    # Extract precision and recall if available
    # Note: These metrics may be 0.0 if model predictions are all one class
    if 'precision' in history.history and len(history.history['precision']) > 0:
        performance_log['final_train_precision'] = float(history.history['precision'][-1])
        if 'val_precision' in history.history and len(history.history['val_precision']) > 0:
            performance_log['final_val_precision'] = float(history.history['val_precision'][-1])
        else:
            performance_log['final_val_precision'] = None
    else:
        performance_log['final_train_precision'] = None
        performance_log['final_val_precision'] = None
    
    if 'recall' in history.history and len(history.history['recall']) > 0:
        performance_log['final_train_recall'] = float(history.history['recall'][-1])
        if 'val_recall' in history.history and len(history.history['val_recall']) > 0:
            performance_log['final_val_recall'] = float(history.history['val_recall'][-1])
        else:
            performance_log['final_val_recall'] = None
    else:
        performance_log['final_train_recall'] = None
        performance_log['final_val_recall'] = None
    
    # Debug: Print available history keys if precision/recall are missing
    if performance_log.get('final_train_precision') is None:
        print(f"Warning: Precision metrics not found in history. Available keys: {list(history.history.keys())}")
    if performance_log.get('final_train_recall') is None:
        print(f"Warning: Recall metrics not found in history. Available keys: {list(history.history.keys())}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"  Duration: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")
    print(f"  Throughput: {throughput:.2f} images/second")
    print(f"\n  Final Training Metrics:")
    print(f"    Accuracy:  {performance_log['final_train_accuracy']:.4f}")
    if 'final_train_precision' in performance_log:
        print(f"    Precision: {performance_log['final_train_precision']:.4f}")
    if 'final_train_recall' in performance_log:
        print(f"    Recall:    {performance_log['final_train_recall']:.4f}")
    print(f"    Loss:      {performance_log['final_train_loss']:.4f}")
    print(f"\n  Final Validation Metrics:")
    print(f"    Accuracy:  {performance_log['final_val_accuracy']:.4f}")
    if 'final_val_precision' in performance_log:
        print(f"    Precision: {performance_log['final_val_precision']:.4f}")
    if 'final_val_recall' in performance_log:
        print(f"    Recall:    {performance_log['final_val_recall']:.4f}")
    print(f"    Loss:      {performance_log['final_val_loss']:.4f}")
    print("="*60 + "\n")
    
    return model, history, performance_log


def save_model_and_history(model, history, performance_log, csv_path, model_dir='models'):
    """
    Save trained model, history, and performance log.
    
    Args:
        model: Trained Keras model
        history: Training history
        performance_log: Performance metrics dictionary
        csv_path: Path to CSV file (for naming)
        model_dir: Directory to save model
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate model filename from CSV basename and timestamp
    csv_basename = Path(csv_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{csv_basename}_{timestamp}.keras"
    model_path = os.path.join(model_dir, model_filename)
    
    # Save model
    print(f"Saving model to: {model_path}")
    model.save(model_path)
    print(f"✓ Model saved successfully")
    
    # Save training history
    history_path = os.path.join(model_dir, f"{csv_basename}_{timestamp}_history.json")
    history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"✓ Training history saved to: {history_path}")
    
    # Save performance log
    performance_log_path = os.path.join(model_dir, f"{csv_basename}_{timestamp}_performance.json")
    with open(performance_log_path, 'w') as f:
        json.dump(performance_log, f, indent=2)
    print(f"✓ Performance log saved to: {performance_log_path}")
    
    return model_path, history_path, performance_log_path


def main():
    """Main function to handle command-line arguments and orchestrate training."""
    parser = argparse.ArgumentParser(
        description="Train CNN model for northern lights detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 train_northern_lights_cnn.py --csv data/frames_x1_local_paths.csv
  python3 train_northern_lights_cnn.py --csv data/frames_x1_local_paths.csv --batch-size 64 --epochs 100
        """
    )
    
    parser.add_argument('--csv', required=True, help='Path to CSV file with local_path and label columns')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--train-split', type=float, default=0.8, help='Train/validation split ratio (default: 0.8)')
    parser.add_argument('--model-dir', default='models', help='Directory to save model (default: models)')
    parser.add_argument('--performance-log-dir', default='performance_logs', 
                       help='Directory to save performance logs (default: performance_logs)')
    
    args = parser.parse_args()
    
    try:
        # Configure GPU
        print("Configuring GPU...")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth to avoid OOM
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ Found {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"Warning: {e}")
        else:
            print("⚠ No GPU found, using CPU")
        
        # Load data from CSV
        paths, labels, original_labels = load_data_from_csv(args.csv)
        
        # Show label distribution
        unique_labels, counts = np.unique(original_labels, return_counts=True)
        print("\nOriginal Label Distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"  {label}: {count}")
        
        binary_counts = np.bincount(labels)
        print("\nBinary Label Distribution:")
        print(f"  Class 0 (no northern lights): {binary_counts[0]}")
        print(f"  Class 1 (northern lights): {binary_counts[1]}")
        
        # Detect image size from first image
        image_size = detect_image_size(paths)
        
        # Train/validation split on paths and labels
        split_idx = int(len(paths) * args.train_split)
        train_paths = paths[:split_idx]
        train_labels = labels[:split_idx]
        val_paths = paths[split_idx:]
        val_labels = labels[split_idx:]
        
        print(f"\nData Split:")
        print(f"  Training: {len(train_paths)} samples")
        print(f"  Validation: {len(val_paths)} samples")
        print(f"  Using data generators (memory-efficient for large datasets)")
        
        # Create data generators
        print("\nCreating data generators...")
        train_generator = DataGenerator(
            train_paths, train_labels, image_size,
            batch_size=args.batch_size,
            shuffle=True,
            augment=True  # Enable augmentation for training
        )
        
        val_generator = DataGenerator(
            val_paths, val_labels, image_size,
            batch_size=args.batch_size,
            shuffle=False,
            augment=False  # No augmentation for validation
        )
        
        # Build model
        print("\nBuilding CNN model...")
        input_shape = (image_size[1], image_size[0], 3)  # (height, width, channels)
        model = build_cnn_model(input_shape)
        
        print("\nModel Architecture:")
        model.summary()
        
        # Create performance log directory
        os.makedirs(args.performance_log_dir, exist_ok=True)
        
        # Train model
        model, history, performance_log = train_model(
            model, train_generator, val_generator,
            num_train_samples=len(train_paths),
            num_val_samples=len(val_paths),
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            model_dir=args.model_dir
        )
        
        # Save model and history
        model_path, history_path, performance_log_path = save_model_and_history(
            model, history, performance_log, args.csv, args.model_dir
        )
        
        # Move performance log to performance_logs directory
        performance_log_dir = args.performance_log_dir
        os.makedirs(performance_log_dir, exist_ok=True)
        final_perf_log_path = os.path.join(performance_log_dir, os.path.basename(performance_log_path))
        if os.path.exists(performance_log_path):
            import shutil
            shutil.move(performance_log_path, final_perf_log_path)
            print(f"✓ Performance log moved to: {final_perf_log_path}")
        
        print("\n" + "="*60)
        print("✓ Training completed successfully!")
        print("="*60)
        print(f"  Model: {model_path}")
        print(f"  History: {history_path}")
        print(f"  Performance log: {final_perf_log_path}")
        print("="*60)
        
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

