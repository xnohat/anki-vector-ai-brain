# Object Detection on Raspberry Pi - Guide

This project has been updated to replace **nanoowl** (NVIDIA CUDA-only) with Raspberry Pi-compatible alternatives.

## Two Options Available

### Option 1: OWL-ViT with Transformers (Default) ✨

**File:** `src/owl.py`

**Pros:**
- ✅ Zero-shot detection - detect ANY object by name
- ✅ Same capability as original nanoowl
- ✅ No training required
- ✅ Works on CPU (Raspberry Pi compatible)

**Cons:**
- ⚠️ Slower inference (~2-5 seconds per frame on RPi 4)
- ⚠️ Higher memory usage (~1-2GB RAM)
- ⚠️ Larger model download (~500MB)

**Installation:**
```bash
pip install torch torchvision transformers pillow numpy
```

**Usage:** Already configured in `app.py` - no changes needed!

---

### Option 2: YOLOv8 (Faster Alternative) 🚀

**File:** `src/owl_yolo.py`

**Pros:**
- ✅ Much faster inference (~0.1-0.5 seconds per frame on RPi 4)
- ✅ Lower memory usage (~500MB RAM)
- ✅ Smaller model (~6MB)
- ✅ Real-time capable

**Cons:**
- ⚠️ Limited to 80 pre-trained COCO classes
- ⚠️ Cannot detect arbitrary objects
- ⚠️ Not true zero-shot detection

**COCO Classes Available:** person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

**Installation:**
```bash
pip install ultralytics
```

**To Switch to YOLO:**
```bash
# Backup current implementation
mv src/owl.py src/owl_transformers.py

# Use YOLO implementation
mv src/owl_yolo.py src/owl.py
```

---

## Performance Comparison

| Feature | OWL-ViT (Transformers) | YOLOv8 |
|---------|----------------------|--------|
| Speed (RPi 4) | 2-5 sec/frame | 0.1-0.5 sec/frame |
| Memory Usage | 1-2 GB | 500 MB |
| Model Size | ~500 MB | ~6 MB |
| Detection Type | Zero-shot (any object) | Pre-trained (80 classes) |
| Accuracy | High for described objects | High for COCO classes |
| Best For | Flexibility, unique objects | Speed, common objects |

---

## Recommendations

### Use OWL-ViT (Default) if:
- You need to detect unusual/specific objects (e.g., "toy robot", "blue cup")
- Speed is not critical
- You have a Raspberry Pi 4 with 4GB+ RAM

### Use YOLOv8 if:
- You need real-time detection
- You only need common objects (people, cars, animals, etc.)
- You have limited RAM (works on RPi 3B+)
- Speed is more important than flexibility

---

## Testing

Test the object detection with the provided test script:

```bash
python3 scripts/test_detection.py
```

This will:
1. Load a test image or capture from Vector's camera
2. Run object detection
3. Save the annotated result
4. Print performance metrics

---

## Optimization Tips

### For OWL-ViT:
- Use smaller images (resize to 640x480 before detection)
- Reduce the number of query objects in prompt
- Consider using `google/owlvit-base-patch16` for slightly better speed

### For YOLOv8:
- Use `yolov8n.pt` (nano) for fastest speed
- Use `yolov8s.pt` (small) for better accuracy
- Adjust confidence threshold (higher = faster)

---

## Troubleshooting

### Out of Memory Error
- Close other applications
- Use YOLOv8 instead of OWL-ViT
- Reduce image size
- Add swap space: `sudo dphys-swapfile swapsize=2048`

### Slow Inference
- First inference is always slower (model loading)
- Subsequent inferences are cached and faster
- Consider switching to YOLOv8
- Overclock your Raspberry Pi (if comfortable)

### Model Download Issues
- Models download automatically on first run
- Requires internet connection
- Downloads are cached in `~/.cache/huggingface/` (OWL-ViT) or `~/.ultralytics/` (YOLO)

