# Memory Analysis Report: Batch Size vs Image Size

## Executive Summary

This report analyzes memory requirements for training a CNN model on two different dataset configurations:
1. **No Resize Dataset**: 1920×1080 images (~1.5MB file size)
2. **10x Resize Dataset**: ~256×144 images (~25KB file size)

The analysis explains why batch size 256 works optimally for resized images but batch size 1 fails for full-resolution images.

---

## 1. Memory Estimation Calculations

### 1.1 Dataset Configuration 1: No Resize (1920×1080, ~1.5MB)

#### Image Memory Calculation:
```
Image dimensions: 1920 × 1080 × 3 channels (RGB)
Memory per image (float32): 1920 × 1080 × 3 × 4 bytes
                           = 6,220,800 × 4 bytes
                           = 24,883,200 bytes
                           = 23.73 MB per image
```

#### Model Architecture Memory (After 4 Conv+Pool Layers):
```
Input: (1920, 1080, 3)
After Conv1+Pool: (960, 540, 32)
After Conv2+Pool: (480, 270, 64)
After Conv3+Pool: (240, 135, 128)
After Conv4+Pool: (120, 67, 128) ≈ (120, 68, 128) for calculation

Flatten output: 120 × 68 × 128 = 1,044,480 elements
Dense(512) layer: 1,044,480 × 512 = 535,004,160 parameters
```

#### Total Memory Breakdown for Batch Size 1:
```
1. Input batch (1 image):
   1 × 23.73 MB = 23.73 MB

2. Model weights:
   Dense layer: 535M × 4 bytes = 2,140 MB ≈ 2.09 GB
   Conv layers: ~50-100 MB
   Total model weights: ~2.2 GB

3. Gradients (same size as weights):
   ~2.2 GB

4. Optimizer states (Adam stores momentum + variance):
   Momentum: ~2.2 GB
   Variance: ~2.2 GB
   Total optimizer: ~4.4 GB

5. Forward pass activations:
   - Input: 23.73 MB
   - Conv activations: ~500 MB - 1 GB
   - Dense input (Flatten): ~4 MB
   - Dense output: ~2 MB
   Total activations: ~600 MB - 1.2 GB

6. Backward pass (gradients computation):
   ~600 MB - 1.2 GB (similar to forward)

7. CUDA/TensorFlow overhead:
   ~500 MB - 1 GB

TOTAL MEMORY REQUIRED:
= 23.73 MB (input)
  + 2.2 GB (model weights)
  + 2.2 GB (gradients)
  + 4.4 GB (optimizer)
  + 1.2 GB (activations)
  + 1.2 GB (backward)
  + 1 GB (overhead)
= ~13.2 GB

Available GPU: 12 GB
RESULT: OUT OF MEMORY (even with batch size 1)
```

---

### 1.2 Dataset Configuration 2: 10x Resize (~256×144, ~25KB)

#### Image Memory Calculation:
```
Image dimensions: ~256 × 144 × 3 channels (RGB)
Memory per image (float32): 256 × 144 × 3 × 4 bytes
                           = 110,592 × 4 bytes
                           = 442,368 bytes
                           = 0.42 MB per image
```

#### Model Architecture Memory (After 4 Conv+Pool Layers):
```
Input: (256, 144, 3)
After Conv1+Pool: (128, 72, 32)
After Conv2+Pool: (64, 36, 64)
After Conv3+Pool: (32, 18, 128)
After Conv4+Pool: (16, 9, 128)

Flatten output: 16 × 9 × 128 = 18,432 elements
Dense(512) layer: 18,432 × 512 = 9,437,184 parameters
```

#### Total Memory Breakdown for Batch Size 256:
```
1. Input batch (256 images):
   256 × 0.42 MB = 107.52 MB

2. Model weights:
   Dense layer: 9.4M × 4 bytes = 37.6 MB
   Conv layers: ~10-20 MB
   Total model weights: ~50 MB

3. Gradients:
   ~50 MB

4. Optimizer states (Adam):
   Momentum: ~50 MB
   Variance: ~50 MB
   Total optimizer: ~100 MB

5. Forward pass activations:
   - Input: 107.52 MB
   - Conv activations: ~200-400 MB
   - Dense input (Flatten): ~70 MB
   - Dense output: ~0.5 MB
   Total activations: ~400-600 MB

6. Backward pass:
   ~400-600 MB

7. CUDA/TensorFlow overhead:
   ~500 MB - 1 GB

TOTAL MEMORY REQUIRED:
= 107.52 MB (input)
  + 50 MB (model weights)
  + 50 MB (gradients)
  + 100 MB (optimizer)
  + 600 MB (activations)
  + 600 MB (backward)
  + 1 GB (overhead)
= ~3.0 GB

Available GPU: 12 GB
RESULT: FITS EASILY (with batch size 256)
```

---

## 2. Why Batch Size 256 Gives Best Results for 10x Resize Dataset

### 2.1 Memory Efficiency
- **Small image size** (0.42 MB per image) allows processing 256 images with only ~108 MB input memory
- **Small model** (18K elements after Flatten) results in only 9.4M parameters in Dense layer
- **Total memory usage**: ~3 GB, leaving 9 GB headroom for larger batches if needed

### 2.2 GPU Utilization
- **Optimal batch size** balances:
  - **Too small (e.g., 8-32)**: GPU underutilized, more overhead from frequent kernel launches
  - **Too large (e.g., 512+)**: Diminishing returns, may cause memory fragmentation
  - **256**: Sweet spot for RTX 4070, maximizes GPU throughput

### 2.3 Training Efficiency
- **Fewer batches per epoch**: 18,381 samples ÷ 256 = 72 batches (vs 575 with batch size 32)
- **Less overhead**: Fewer data transfers between CPU/GPU
- **Better gradient estimates**: Larger batches provide more stable gradients
- **Faster training**: Higher throughput (images/second) as seen in your performance logs

### 2.4 Performance Metrics from Your Data
Based on your performance logs:
- **Batch size 256**: ~245 seconds, 3750 images/sec (best throughput)
- **Batch size 128**: ~252-304 seconds, 3019-3642 images/sec
- **Batch size 512**: ~395-414 seconds, 2216-2325 images/sec (slower due to overhead)

**Conclusion**: Batch size 256 maximizes GPU utilization while staying within memory limits for resized images.

---

## 3. Why Batch Size 1 Fails for No Resize Dataset

### 3.1 The Problem: Model Architecture, Not Batch Size

The critical issue is the **Flatten() → Dense(512)** layer combination:

```
For 1920×1080 images:
Flatten output: 1,044,480 elements
Dense(512) parameters: 1,044,480 × 512 = 535,004,160 parameters
Memory for this layer alone: 535M × 4 bytes = 2.14 GB

With gradients + optimizer: 2.14 GB × 3 = 6.42 GB
```

### 3.2 Memory Breakdown (Batch Size 1):

| Component | Memory | Notes |
|-----------|--------|-------|
| Input batch (1 image) | 23.73 MB | Negligible |
| Model weights | 2.2 GB | Dense layer dominates |
| Gradients | 2.2 GB | Same as weights |
| Optimizer (Adam) | 4.4 GB | Momentum + variance |
| Forward activations | 1.2 GB | Through all layers |
| Backward pass | 1.2 GB | Gradient computation |
| CUDA overhead | 1.0 GB | Framework overhead |
| **TOTAL** | **~13.2 GB** | **Exceeds 12 GB GPU!** |

### 3.3 Why Reducing Batch Size Doesn't Help

**Batch size only affects input memory**, not model memory:

| Batch Size | Input Memory | Model Memory | Total | Result |
|------------|--------------|--------------|-------|--------|
| 1 | 23.73 MB | 11.0 GB | 13.2 GB | ❌ OOM |
| 2 | 47.46 MB | 11.0 GB | 13.2 GB | ❌ OOM |
| 4 | 94.92 MB | 11.0 GB | 13.2 GB | ❌ OOM |
| 8 | 189.84 MB | 11.0 GB | 13.2 GB | ❌ OOM |

**Key Insight**: Model memory (11 GB) is fixed regardless of batch size. Only input memory changes, which is negligible compared to model memory.

### 3.4 The Root Cause

The model architecture is **not scalable** to high-resolution images:
- `Flatten()` creates a huge feature vector (1,044,480 elements)
- `Dense(512)` layer has 535M parameters
- This design works for small images but fails for large ones

---

## 4. Solutions for No Resize Dataset

### Solution 1: Use GlobalAveragePooling2D (Recommended)
```
Replace:
  Flatten() → Dense(512)  [535M parameters, 2.14 GB]

With:
  GlobalAveragePooling2D() → Dense(512)  [65K parameters, 0.25 MB]

Memory reduction: 2.14 GB → 0.25 MB (99.99% reduction)
Result: Batch size 8-16 should work
```

### Solution 2: Reduce Image Resolution
```
Resize to 960×540 or 640×360 before training
Reduces Flatten output significantly
```

### Solution 3: Enable Mixed Precision
```
Use float16 instead of float32
Reduces memory by ~50%
May allow batch size 2-4
```

### Solution 4: Reduce Model Capacity
```
Reduce Dense layer: 512 → 256 or 128
Reduces parameters but may hurt accuracy
```

---

## 5. Summary Table

| Dataset | Image Size | File Size | Batch Size | Input Mem | Model Mem | Total Mem | Result |
|---------|-----------|-----------|------------|-----------|-----------|-----------|--------|
| 10x Resize | 256×144 | 25KB | 256 | 108 MB | 200 MB | ~3 GB | ✅ Works |
| 10x Resize | 256×144 | 25KB | 128 | 54 MB | 200 MB | ~2.5 GB | ✅ Works |
| 10x Resize | 256×144 | 25KB | 512 | 216 MB | 200 MB | ~3.5 GB | ✅ Works |
| No Resize | 1920×1080 | 1.5MB | 1 | 24 MB | 11 GB | ~13.2 GB | ❌ OOM |
| No Resize | 1920×1080 | 1.5MB | 2 | 48 MB | 11 GB | ~13.2 GB | ❌ OOM |
| No Resize | 1920×1080 | 1.5MB | 4 | 96 MB | 11 GB | ~13.2 GB | ❌ OOM |

---

## 6. Key Findings

1. **Batch size 256 is optimal for resized images** because:
   - Small images (0.42 MB) allow large batches
   - Small model (9.4M parameters) fits easily
   - Maximizes GPU utilization and training throughput

2. **Batch size 1 fails for full-resolution images** because:
   - Model architecture creates 535M parameters (2.14 GB)
   - With gradients and optimizer: ~11 GB fixed cost
   - Input memory (24 MB) is negligible
   - Total exceeds 12 GB GPU memory

3. **The bottleneck is model architecture, not batch size**:
   - Flatten() on large images creates huge feature vectors
   - Dense layer scales quadratically with image size
   - Solution: Use GlobalAveragePooling2D() instead

---

## 7. Recommendations

1. **For 10x Resize Dataset**: Continue using batch size 256 (optimal)
2. **For No Resize Dataset**: 
   - Modify model to use `GlobalAveragePooling2D()` instead of `Flatten()`
   - Or resize images to 960×540 or smaller
   - Or enable mixed precision training
3. **Future Architecture**: Design models that scale better with image resolution using global pooling or attention mechanisms

---

## Appendix: Memory Calculation Formulas

### Per Image Memory:
```
Memory = width × height × channels × bytes_per_float
       = W × H × 3 × 4 bytes (for RGB float32)
```

### Model Memory (Dense Layer):
```
Parameters = (flatten_size) × (dense_units)
Memory = parameters × 4 bytes × 3 (weights + gradients + optimizer)
```

### Batch Memory:
```
Batch Memory = batch_size × per_image_memory × multiplier
Multiplier ≈ 4-5 (accounts for forward + backward + activations)
```

### Total Training Memory:
```
Total = Input_Batch_Memory + Model_Memory + Activations + Overhead
```

