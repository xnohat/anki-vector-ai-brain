# Quick Start Guide - Raspberry Pi Object Detection

## 🚀 Installation (One Command)

```bash
./scripts/install_rpi.sh
```

Follow the prompts to choose between OWL-ViT (flexible) or YOLOv8 (fast).

---

## 📦 Manual Installation

### Option 1: OWL-ViT (Zero-shot Detection)

```bash
# Install dependencies
pip install torch torchvision transformers pillow numpy

# No code changes needed - already configured!
```

### Option 2: YOLOv8 (Faster, 80 classes)

```bash
# Install dependencies
pip install ultralytics pillow numpy

# Switch implementation
mv src/owl.py src/owl_transformers.py
cp src/owl_yolo.py src/owl.py
```

---

## ✅ Test Installation

```bash
python3 scripts/test_detection.py
```

This will test detection and save results to `output/` folder.

---

## 🎯 Usage Examples

### Basic Detection

```python
from owl import HootHoot

# Initialize
detector = HootHoot()

# Detect objects
output, image = detector.predict(
    image=my_image,
    prompt="[person, car, dog]",
    threshold=0.1
)

# Results
print(f"Found {output['count']} objects")
for det in output['detections']:
    print(f"  {det['label']}: {det['score']:.2%}")
```

### With Vector Robot Camera

```python
# In your Vector app (see app.py for full example)
frame = robot_data.get_pil_frame()
output, annotated = detector.predict(frame, "[person, toy]")
```

---

## 🎛️ Configuration Options

### OWL-ViT

```python
detector = HootHoot(
    model="google/owlvit-base-patch32",  # or owlvit-base-patch16
    device="cpu"
)
```

### YOLOv8

```python
detector = HootHoot(
    model="yolov8n.pt",  # n=nano, s=small, m=medium
    device="cpu"
)
```

---

## 💡 Tips & Tricks

### Improve Speed

1. **Reduce image size:**
   ```python
   image = image.resize((640, 480))
   ```

2. **Use YOLOv8 instead of OWL-ViT**

3. **Increase threshold:**
   ```python
   output, img = detector.predict(image, prompt, threshold=0.3)
   ```

### Improve Memory Usage

1. **Use YOLOv8 (uses less RAM)**

2. **Add swap space:**
   ```bash
   sudo dphys-swapfile swapsize=2048
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

### Available Classes (YOLOv8 only)

```
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird,
cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack,
umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball,
kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket,
bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple,
sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair,
couch, potted plant, bed, dining table, toilet, tv, laptop, mouse,
remote, keyboard, cell phone, microwave, oven, toaster, sink,
refrigerator, book, clock, vase, scissors, teddy bear, hair drier,
toothbrush
```

---

## 🔧 Common Issues

### "Out of memory"
→ Use YOLOv8 or add swap space

### "Too slow"
→ Switch to YOLOv8 or reduce image size

### "No objects detected"
→ Lower threshold or check prompt format

### "Module not found"
→ Run `pip install -r requirements.txt`

---

## 📊 Quick Comparison

|  | **OWL-ViT** | **YOLOv8** |
|---|---|---|
| Speed (RPi 4) | 2-5 sec | 0.1-0.5 sec |
| Flexibility | Any object | 80 classes only |
| Memory | 1-2 GB | 500 MB |
| Model Size | 500 MB | 6 MB |
| Best For | Unusual objects | Common objects |

---

## 🔗 More Information

- **Full Guide:** `OBJECT_DETECTION_GUIDE.md`
- **Migration Info:** `MIGRATION_SUMMARY.md`
- **Test Script:** `scripts/test_detection.py`
- **Install Script:** `scripts/install_rpi.sh`

---

## 🏃 Run the App

```bash
python3 app.py
```

Enjoy your Raspberry Pi-powered Vector robot with AI vision! 🤖✨

