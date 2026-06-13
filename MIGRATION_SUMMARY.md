# Migration from nanoowl to Raspberry Pi Compatible Libraries

## Summary of Changes

This document summarizes the changes made to replace **nanoowl** (NVIDIA CUDA/TensorRT only) with **Raspberry Pi compatible alternatives**.

---

## What Was Changed

### 1. **Main Object Detection Module** (`src/owl.py`)

**Before:**
- Used nanoowl library (requires NVIDIA GPU)
- Required CUDA and TensorRT
- Built optimized engines for inference
- Only worked on Jetson/NVIDIA hardware

**After:**
- Uses Hugging Face Transformers library
- Runs on CPU (Raspberry Pi compatible)
- Maintains zero-shot detection capability
- Same API interface for easy migration

**Key Changes:**
```python
# Old (nanoowl)
from nanoowl.tree_predictor import OwlPredictor, TreePredictor, Tree

# New (transformers)
from transformers import OwlViTProcessor, OwlViTForObjectDetection
```

### 2. **Alternative Implementation** (`src/owl_yolo.py`)

**New File:**
- Provides YOLOv8-based object detection
- Much faster than OWL-ViT on Raspberry Pi
- Limited to 80 COCO classes (not zero-shot)
- Optional replacement for speed-critical applications

### 3. **Dependencies** (`requirements.txt`)

**Added:**
```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
Pillow>=9.0.0
numpy>=1.21.0
```

**Removed:**
- nanoowl (no longer needed)
- CUDA/TensorRT dependencies

### 4. **Documentation Updates**

**New Files:**
- `OBJECT_DETECTION_GUIDE.md` - Comprehensive guide for both implementations
- `MIGRATION_SUMMARY.md` - This file
- `scripts/test_detection.py` - Test script for verification
- `scripts/install_rpi.sh` - Automated installation script

**Updated Files:**
- `README.md` - Updated installation instructions

---

## API Compatibility

The new implementation maintains the same API as the original:

```python
from owl import HootHoot

# Initialize detector
detector = HootHoot()

# Perform detection (same as before)
output, annotated_image = detector.predict(
    image=image,
    prompt="[a person, toys]",
    threshold=0.1
)
```

**No changes required in `app.py`!** The migration is transparent.

---

## Performance Comparison

### Original (nanoowl on Jetson Nano)
- Inference: ~50-100ms per frame
- Hardware: Requires NVIDIA GPU
- Memory: ~2GB
- Power: ~10W

### New Option 1 (OWL-ViT on Raspberry Pi 4)
- Inference: ~2-5 seconds per frame
- Hardware: Raspberry Pi CPU
- Memory: ~1-2GB
- Power: ~5W

### New Option 2 (YOLOv8 on Raspberry Pi 4)
- Inference: ~100-500ms per frame
- Hardware: Raspberry Pi CPU
- Memory: ~500MB
- Power: ~5W

---

## Migration Steps

### For Existing Users

1. **Update dependencies:**
   ```bash
   pip uninstall nanoowl  # Remove old dependency
   pip install -r requirements.txt  # Install new dependencies
   ```

2. **No code changes needed!**
   The new `src/owl.py` maintains the same API.

3. **Test the changes:**
   ```bash
   python3 scripts/test_detection.py
   ```

### For New Users

1. **Run the installation script:**
   ```bash
   ./scripts/install_rpi.sh
   ```

2. **Test the installation:**
   ```bash
   python3 scripts/test_detection.py
   ```

3. **Run the application:**
   ```bash
   python3 app.py
   ```

---

## Feature Comparison

| Feature | nanoowl (Old) | OWL-ViT (New) | YOLOv8 (New) |
|---------|---------------|---------------|--------------|
| **Zero-shot Detection** | ✅ Yes | ✅ Yes | ❌ No (80 classes) |
| **Raspberry Pi Support** | ❌ No | ✅ Yes | ✅ Yes |
| **NVIDIA GPU Required** | ✅ Yes | ❌ No | ❌ No |
| **Inference Speed (RPi 4)** | N/A | Slow (2-5s) | Fast (0.1-0.5s) |
| **Model Size** | ~200MB | ~500MB | ~6MB |
| **Memory Usage** | ~2GB | ~1-2GB | ~500MB |
| **Real-time Capable** | ✅ Yes | ❌ No | ✅ Yes |

---

## Troubleshooting

### Issue: Out of Memory Error

**Solution:**
1. Increase swap space:
   ```bash
   sudo dphys-swapfile swapsize=2048
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

2. Or switch to YOLOv8:
   ```bash
   mv src/owl.py src/owl_transformers.py
   cp src/owl_yolo.py src/owl.py
   pip install ultralytics
   ```

### Issue: Slow Inference

**Solution:**
1. Reduce image size before detection
2. Switch to YOLOv8 for faster inference
3. Overclock Raspberry Pi (if comfortable)
4. Use Raspberry Pi 4 with 4GB+ RAM

### Issue: Model Download Fails

**Solution:**
1. Check internet connection
2. Manually download models:
   ```python
   from transformers import OwlViTProcessor, OwlViTForObjectDetection
   OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
   OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
   ```

---

## Future Improvements

Possible enhancements:

1. **ONNX Runtime Support**
   - Convert models to ONNX for faster CPU inference
   - Estimated 2-3x speedup

2. **Model Quantization**
   - Reduce model size and memory usage
   - Trade slight accuracy for speed

3. **Coral TPU Support**
   - Use Google Coral USB Accelerator
   - Enable real-time inference with OWL-ViT

4. **Batch Processing**
   - Process multiple frames in parallel
   - Better throughput for video streams

---

## Support

For issues or questions:

1. Check `OBJECT_DETECTION_GUIDE.md` for detailed information
2. Run the test script: `python3 scripts/test_detection.py`
3. Check memory usage: `free -h`
4. Monitor CPU temperature: `vcgencmd measure_temp`

---

## License

Same as original project (see LICENSE file).

## Credits

- Original project used [nanoowl](https://github.com/NVIDIA-AI-IOT/nanoowl) by NVIDIA
- New implementation uses [Hugging Face Transformers](https://huggingface.co/transformers/)
- Alternative uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

