#!/usr/bin/env python3
"""
PySpark CNN Training Script for Binary Northern Lights Classification

Trains a CNN model to classify images as "northern lights exist" (1) or "not exist" (0)
using PyTorch with distributed training via TorchDistributor.
"""

import argparse
import sys
import time
import logging
import os
import subprocess
import tempfile
import json
from datetime import datetime
from typing import Tuple, List, Iterator, Optional
from collections import defaultdict

import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when
from pyspark.ml.torch.distributor import TorchDistributor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- CONSTANTS ---
IMG_HEIGHT = 192
IMG_WIDTH = 108
IMG_CHANNELS = 3


# --- UTILITY FUNCTIONS ---

def format_duration(seconds: float) -> str:
    """Format seconds into human-readable HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = round(seconds % 60, 2)
    return f"{hours:02}:{minutes:02}:{secs:05.2f}"


def save_model_to_hdfs(model: object, hdfs_path: str):
    """
    Save PyTorch model to HDFS from the driver node. Uses external 'hadoop fs' commands.
    """
    local_temp_dir = f"/tmp/model_{os.getpid()}_{os.urandom(8).hex()}"
    os.makedirs(local_temp_dir, exist_ok=True)
    
    try:
        # Save PyTorch model
        import torch
        model_path = os.path.join(local_temp_dir, "model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_architecture': model.__class__.__name__,
            'img_height': IMG_HEIGHT,
            'img_width': IMG_WIDTH,
            'img_channels': IMG_CHANNELS
        }, model_path)
        logger.info(f"Model saved locally to {model_path}")
        
        hdfs_path_clean = hdfs_path.replace("hdfs://", "")
        
        subprocess.run(['hadoop', 'fs', '-mkdir', '-p', hdfs_path_clean], check=True, stderr=subprocess.PIPE)
        subprocess.run(
            ['hadoop', 'fs', '-copyFromLocal', '-f', model_path, os.path.join(hdfs_path_clean, "model.pt")],
            check=True,
            stderr=subprocess.PIPE
        )
        
        logger.info(f"Model saved to HDFS: {hdfs_path}/model.pt")
        
    except Exception as e:
        logger.error(f"Error saving model to HDFS: {e}")
        raise
    finally:
        if os.path.exists(local_temp_dir):
            try:
                subprocess.run(['rm', '-rf', local_temp_dir], check=True)
            except:
                pass


# --- DRIVER DATA LOGIC (USES SPARK) ---

def load_and_split_data(spark: SparkSession, args: argparse.Namespace) -> Tuple[DataFrame, DataFrame, int, int]:
    """Loads data from Hive, applies label logic, splits into train/val."""
    logger.info("Starting data loading and splitting...")
    df = spark.read.table(args.table_name)
    df = df.withColumn("binary_label", when(col("response").contains("northern light"), 1).otherwise(0))
    df = df.filter(col("file_uri").isNotNull())
    df = df.select(col("file_uri"), col("binary_label"))
    
    logger.info("Label distribution:")
    try:
        df.groupBy("binary_label").count().show()
    except Exception as e:
        logger.warning(f"Could not display label distribution: {e}")
        label_counts = df.groupBy("binary_label").count().collect()
        for row in label_counts:
            logger.info(f"  Label {row['binary_label']}: {row['count']} samples")
    
    splits = df.randomSplit([args.train_val_split, 1.0 - args.train_val_split], seed=42)
    train_df = splits[0].repartition(args.num_partitions)
    val_df = splits[1].repartition(args.num_partitions)
    
    train_count = train_df.count()
    val_count = val_df.count()
    logger.info(f"Train samples: {train_count}, Validation samples: {val_count}")
    
    return train_df, val_df, train_count, val_count


# --- PYTORCH MODEL DEFINITION ---

def build_cnn_model(input_shape: Tuple[int, int, int] = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    """
    Build CNN model matching RTX architecture for binary classification.
    
    Architecture:
    - 4 convolutional blocks (32, 64, 128, 128 filters)
    - Dense layers: Dropout(0.5) -> Dense(512, relu) -> Dropout(0.5) -> Dense(1, sigmoid)
    """
    import torch
    import torch.nn as nn
    
    class NorthernLightsCNN(nn.Module):
        def __init__(self):
            super(NorthernLightsCNN, self).__init__()
            
            # First convolutional block
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            
            # Second convolutional block
            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            
            # Third convolutional block
            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            
            # Fourth convolutional block
            self.conv4 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            
            # Calculate flattened size: (192/16) * (108/16) * 128 = 12 * 6 * 128 = 9216
            self.flatten = nn.Flatten()
            self.dropout1 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 512)
            self.relu = nn.ReLU()
            self.dropout2 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(512, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.flatten(x)
            x = self.dropout1(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            return x
    
    return NorthernLightsCNN()


# --- DISTRIBUTED TRAINING FUNCTION (WORKER) ---

def train_model_fn(
    train_data_path: str,
    val_data_path: str,
    batch_size: int,
    epochs: int,
    learning_rate: float = 0.001
):
    """
    Training function executed by TorchDistributor on each worker.
    
    According to TorchDistributor API, this function should:
    1. Initialize distributed training using torch.distributed.init_process_group()
    2. Get rank/world_size from the distributed environment
    3. Load its shard of data based on rank
    4. Train the model
    
    Args:
        train_data_path: HDFS path to training data manifest (Parquet file)
        val_data_path: HDFS path to validation data manifest (Parquet file)
        batch_size: Batch size per process
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    """
    # Imports MUST be inside the distributed training function
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    from torch.utils.data import Dataset, DataLoader, DistributedSampler
    import subprocess
    import numpy as np
    from PIL import Image
    import tempfile
    import os
    import json
    import logging
    
    # Set up logger for this worker
    worker_logger = logging.getLogger(__name__)
    
    # Initialize distributed training (required by TorchDistributor)
    # TorchDistributor sets up the environment, but we need to call init_process_group
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='gloo')  # Use gloo for CPU
        device = torch.device('cpu')
        torch.set_num_threads(2)  # Limit threads per process
        is_distributed = True
    else:
        rank = 0
        world_size = 1
        device = torch.device('cpu')
        torch.set_num_threads(4)
        is_distributed = False
    
    def load_image_from_hdfs(hdfs_path: str) -> np.ndarray:
        """Load an image from HDFS path and return as numpy array."""
        if hdfs_path.startswith("hdfs://"):
            hdfs_path_clean = hdfs_path.replace("hdfs://", "")
        else:
            hdfs_path_clean = hdfs_path
        
        # Create unique temp file name to avoid conflicts
        import uuid
        local_temp_file = os.path.join(tempfile.gettempdir(), f"img_{rank}_{uuid.uuid4().hex}.png")
        
        # Remove file if it exists (shouldn't happen with UUID, but just in case)
        if os.path.exists(local_temp_file):
            os.remove(local_temp_file)
        
        try:
            # Copy from HDFS to local
            result = subprocess.run(
                ['hadoop', 'fs', '-copyToLocal', hdfs_path_clean, local_temp_file],
                check=True,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True
            )
            
            # Load and process image
            img = Image.open(local_temp_file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if img.size != (IMG_WIDTH, IMG_HEIGHT):
                img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
            
            img_array = np.array(img, dtype=np.float32) / 255.0
            return img_array
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else "Unknown error"
            raise Exception(f"Error loading image from {hdfs_path}: {error_msg}")
        except Exception as e:
            raise Exception(f"Error loading image from {hdfs_path}: {e}")
        finally:
            if os.path.exists(local_temp_file):
                try:
                    os.remove(local_temp_file)
                except:
                    pass
    
    # Load data manifest from HDFS
    def load_data_manifest(hdfs_path: str) -> List[Tuple[str, int]]:
        """Load data manifest (JSON lines) from HDFS."""
        # Clean HDFS path
        if hdfs_path.startswith("hdfs://"):
            hdfs_path_clean = hdfs_path.replace("hdfs://", "")
        else:
            hdfs_path_clean = hdfs_path
        
        # Verify file exists first
        check_result = subprocess.run(
            ['hadoop', 'fs', '-test', '-e', hdfs_path_clean],
            capture_output=True
        )
        if check_result.returncode != 0:
            raise FileNotFoundError(f"Manifest file not found in HDFS: {hdfs_path_clean}")
        
        # Create a unique temp file name to avoid conflicts
        import uuid
        local_temp_file = os.path.join(tempfile.gettempdir(), f"manifest_{rank}_{uuid.uuid4().hex}.jsonl")
        
        # Remove file if it exists (shouldn't happen with UUID, but just in case)
        if os.path.exists(local_temp_file):
            os.remove(local_temp_file)
        
        try:
            # Copy from HDFS to local
            result = subprocess.run(
                ['hadoop', 'fs', '-copyToLocal', hdfs_path_clean, local_temp_file],
                check=True,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True
            )
            
            worker_logger.info(f"Loaded manifest from {hdfs_path_clean} (rank {rank})")
            
            # Read and parse JSON lines
            data = []
            with open(local_temp_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            item = json.loads(line)
                            data.append((item['file_uri'], item['binary_label']))
                        except json.JSONDecodeError as e:
                            worker_logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                            continue
            
            worker_logger.info(f"Loaded {len(data)} samples from manifest (rank {rank})")
            return data
            
        except subprocess.CalledProcessError as e:
            worker_logger.error(f"Error copying manifest from HDFS: {e.stderr}")
            raise
        finally:
            if os.path.exists(local_temp_file):
                try:
                    os.remove(local_temp_file)
                except:
                    pass
    
    # Load all data (each worker will get its shard via DistributedSampler)
    train_data_list = load_data_manifest(train_data_path)
    val_data_list = load_data_manifest(val_data_path)
    
    class HDFSDataset(Dataset):
        """PyTorch Dataset for loading images from HDFS."""
        def __init__(self, data_list: List[Tuple[str, int]]):
            self.data = data_list
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            file_uri, label = self.data[idx]
            img_array = load_image_from_hdfs(file_uri)
            # Convert to CHW format (PyTorch expects channels first)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
            label_tensor = torch.tensor(label, dtype=torch.float32)
            return img_tensor, label_tensor
    
    # Create datasets
    train_dataset = HDFSDataset(train_data_list)
    val_dataset = HDFSDataset(val_data_list)
    
    # Create distributed samplers if multi-process
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=0,  # No subprocess workers (we're already distributed)
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Build model
    model = build_cnn_model()
    model = model.to(device)
    
    # Wrap with DDP if multi-process
    if is_distributed:
        model = DDP(model)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        # Logging
        if rank == 0:  # Only log from rank 0
            worker_logger.info(
                f"Epoch [{epoch+1}/{epochs}] - "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model state
            if rank == 0:
                best_model_state = model.module.state_dict() if is_distributed else model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if rank == 0:
                    worker_logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Cleanup distributed training
    if is_distributed:
        dist.destroy_process_group()
    
    # Return model (unwrap DDP if needed) - only rank 0 returns
    if rank == 0:
        return model.module if is_distributed else model
    else:
        return None


# --- MAIN DRIVER FUNCTION ---

def main():
    """Main function to parse args and run the training process."""
    parser = argparse.ArgumentParser(
        description='Train CNN model for binary northern lights classification using PyTorch',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--table_name', type=str, required=True)
    parser.add_argument('--model_save_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--num_partitions', type=int, default=20)
    parser.add_argument('--spark_default_parallelism', type=int, default=20)
    parser.add_argument('--spark_shuffle_partitions', type=int, default=20)
    parser.add_argument('--train_val_split', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_executors', type=int, default=10,
                        help='Number of Spark executors to use as training workers.')

    args = parser.parse_args()
    
    if args.model_name:
        model_name = args.model_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"cnn_binary_{timestamp}"
    
    if args.model_save_path.endswith('/'):
        full_model_path = args.model_save_path + model_name
    else:
        full_model_path = args.model_save_path + '/' + model_name
    
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Starting CNN Training (CPU-Only, PyTorch with TorchDistributor)")
    logger.info(f"Image Dimensions: {IMG_HEIGHT}x{IMG_WIDTH}x{IMG_CHANNELS}")
    logger.info("=" * 60)
    
    spark = SparkSession.builder \
        .appName("CNN Binary Classification Training (PyTorch)") \
        .config("spark.default.parallelism", str(args.spark_default_parallelism)) \
        .config("spark.sql.shuffle.partitions", str(args.spark_shuffle_partitions)) \
        .enableHiveSupport() \
        .getOrCreate()
    
    try:
        # Load and split data using the SparkSession
        train_df, val_df, train_count, val_count = load_and_split_data(spark, args)

        training_start = time.time()

        # Save data manifests to HDFS as JSON lines (each worker will read them)
        # Use a unique directory to avoid conflicts
        manifest_id = f"{os.getpid()}_{int(time.time())}"
        hdfs_manifest_dir = f"/data/training_manifests/{manifest_id}"
        hdfs_train_manifest = f"{hdfs_manifest_dir}/train_manifest.jsonl"
        hdfs_val_manifest = f"{hdfs_manifest_dir}/val_manifest.jsonl"
        
        # Create local temp files
        local_train_manifest = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        local_val_manifest = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        
        try:
            logger.info("Creating data manifests...")
            # Write training manifest
            train_count = 0
            for row in train_df.collect():
                json.dump({'file_uri': row.file_uri, 'binary_label': row.binary_label}, local_train_manifest)
                local_train_manifest.write('\n')
                train_count += 1
            
            # Write validation manifest
            val_count = 0
            for row in val_df.collect():
                json.dump({'file_uri': row.file_uri, 'binary_label': row.binary_label}, local_val_manifest)
                local_val_manifest.write('\n')
                val_count += 1
            
            local_train_manifest.close()
            local_val_manifest.close()
            
            logger.info(f"Created local manifests: {train_count} train, {val_count} val samples")
            
            # Create HDFS directory
            subprocess.run(['hadoop', 'fs', '-mkdir', '-p', hdfs_manifest_dir], check=True, 
                         stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            
            # Upload manifests to HDFS
            logger.info(f"Uploading manifests to HDFS: {hdfs_manifest_dir}")
            result = subprocess.run(['hadoop', 'fs', '-put', '-f', local_train_manifest.name, hdfs_train_manifest], 
                                  check=True, capture_output=True, text=True)
            logger.info(f"Uploaded train manifest: {result.stdout}")
            
            result = subprocess.run(['hadoop', 'fs', '-put', '-f', local_val_manifest.name, hdfs_val_manifest], 
                                  check=True, capture_output=True, text=True)
            logger.info(f"Uploaded val manifest: {result.stdout}")
            
            # Verify files exist in HDFS
            result = subprocess.run(['hadoop', 'fs', '-ls', hdfs_train_manifest], 
                                  check=True, capture_output=True, text=True)
            logger.info(f"Verified train manifest exists: {result.stdout}")
            
            result = subprocess.run(['hadoop', 'fs', '-ls', hdfs_val_manifest], 
                                  check=True, capture_output=True, text=True)
            logger.info(f"Verified val manifest exists: {result.stdout}")
            
            logger.info(f"Data manifests saved to HDFS: {hdfs_train_manifest}, {hdfs_val_manifest}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error uploading manifests: {e.stderr}")
            raise
        finally:
            # Clean up local files
            if os.path.exists(local_train_manifest.name):
                os.remove(local_train_manifest.name)
            if os.path.exists(local_val_manifest.name):
                os.remove(local_val_manifest.name)
        
        # Initialize TorchDistributor
        distributor = TorchDistributor(
            num_processes=args.num_executors,
            local_mode=False,  # Distributed mode (multi-node)
            use_gpu=False  # CPU-only training
        )
        
        # Run distributed training
        # According to TorchDistributor API, pass args/kwargs directly to the function
        trained_model = distributor.run(
            train_model_fn,
            hdfs_train_manifest,  # train_data_path
            hdfs_val_manifest,    # val_data_path
            args.batch_size,      # batch_size
            args.epochs,           # epochs
            args.learning_rate     # learning_rate
        )

        training_end = time.time()
        training_duration = training_end - training_start
        logger.info(f"Distributed training completed in {format_duration(training_duration)}")
        
        # Save Model (only rank 0 returns a model)
        if trained_model is not None:
            logger.info(f"Saving trained model to HDFS: {full_model_path}")
            save_model_to_hdfs(trained_model, full_model_path)
        else:
            logger.warning("No model returned from training (non-rank-0 worker?)")
        
        # Final Report
        end_time = time.time()
        total_duration = end_time - start_time
        logger.info("=" * 60)
        logger.info("Training Complete! ✅")
        logger.info(f"Total job time: {format_duration(total_duration)}")
        logger.info(f"Model saved to: {full_model_path}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
