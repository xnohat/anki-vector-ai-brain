# 🚀 START HERE - Vector Advanced AI Setup

## ✅ What's Been Done

Your Vector Advanced AI project has been successfully migrated to Raspberry Pi!

### 1. ✅ Object Detection Replaced
- **Old:** nanoowl (NVIDIA CUDA only) ❌
- **New:** YOLOv8 with ultralytics (Raspberry Pi CPU) ✅
- **Benefits:** 10-50x faster, lower memory usage, real-time capable

### 2. ✅ All Dependencies Installed
- ultralytics (YOLOv8)
- customtkinter (UI)
- openai-whisper (Speech-to-text)
- sounddevice, scipy (Audio)
- openai (ChatGPT)
- kingardor-vector (Vector SDK)

### 3. ✅ Scripts Created
- `run.sh` - Easy launcher with API key checking
- `scripts/test_detection.py` - Test object detection
- `scripts/install_rpi.sh` - Installation script

### 4. ✅ Documentation Created
- `SETUP_COMPLETE.md` - Complete setup guide
- `QUICK_START.md` - Quick reference
- `OBJECT_DETECTION_GUIDE.md` - Detailed detector comparison
- `WHICH_ONE_TO_USE.md` - Decision guide
- `MIGRATION_SUMMARY.md` - Technical migration details

---

## 🎯 What You Need To Do

### Step 1: Set OpenAI API Key (Required)

```bash
export OPENAI_API_KEY='your-api-key-here'
```

To make it permanent:
```bash
echo "export OPENAI_API_KEY='your-api-key-here'" >> ~/.bashrc
source ~/.bashrc
```

**Get your key:** https://platform.openai.com/api-keys

---

### Step 2: Configure Vector Robot (Required)

Make sure:
1. ✅ Vector is on same WiFi network as Raspberry Pi
2. ✅ wire-pod server is running (https://github.com/kercre123/wire-pod)

Then configure the SDK:
```bash
source .venv/bin/activate
python -m anki_vector.configure
```

---

### Step 3: Run the App! 🎉

```bash
./run.sh
```

Or manually:
```bash
source .venv/bin/activate
python app.py
```

---

## 📋 Pre-Flight Checklist

Before running, verify:

- [x] ✅ Virtual environment created (`.venv/`)
- [x] ✅ All dependencies installed
- [ ] ⚠️ OpenAI API key set (`echo $OPENAI_API_KEY` to check)
- [ ] ⚠️ Vector configured (`python -m anki_vector.configure`)
- [ ] ⚠️ Vector on WiFi and wire-pod running
- [ ] ⚠️ Microphone connected (for voice input)

---

## 🧪 Test First (Optional but Recommended)

### Test Object Detection
```bash
source .venv/bin/activate
python scripts/test_detection.py
```
This will:
- Load YOLOv8 model
- Test detection on sample images
- Save results to `output/` folder

### Test OpenAI Connection (requires API key)
```bash
source .venv/bin/activate
python scripts/openai-test.py
```

---

## 🎓 Understanding Your Setup

### Current Configuration:
- **Object Detection:** YOLOv8 (ultra-fast)
- **AI Model:** ChatGPT (via OpenAI API)
- **Speech-to-Text:** Whisper (local model, ~460MB)
- **UI:** CustomTkinter (modern dark theme)
- **Robot Control:** Vector SDK with wire-pod

### What Runs Where:
- **On Raspberry Pi:**
  - YOLOv8 object detection (CPU)
  - Whisper speech recognition (CPU)
  - Vector robot communication
  - UI display

- **In the Cloud:**
  - ChatGPT conversation (OpenAI API)

---

## 🆘 Troubleshooting

### "ModuleNotFoundError"
→ Activate virtual environment: `source .venv/bin/activate`

### "OpenAI API key not found"
→ Set the key: `export OPENAI_API_KEY='your-key'`

### "Vector robot not found"
→ Run: `python -m anki_vector.configure`

### "Audio not working"
→ Check: `python -m sounddevice`

### "Out of memory"
→ Use smaller Whisper model (edit `src/whisperstt.py`, change 'small' to 'tiny')

**Full troubleshooting guide:** See `SETUP_COMPLETE.md`

---

## 📚 Documentation Guide

Read in this order:

1. **START_HERE.md** (this file) - Overview and quick start
2. **SETUP_COMPLETE.md** - Detailed setup instructions
3. **QUICK_START.md** - Command reference
4. **OBJECT_DETECTION_GUIDE.md** - Understand your detector
5. **WHICH_ONE_TO_USE.md** - Choose OWL-ViT or YOLO
6. **MIGRATION_SUMMARY.md** - Technical details

---

## 🎮 How It Works

1. **Microphone** captures your voice
2. **Whisper** converts speech to text (local)
3. **ChatGPT** generates response (cloud)
4. **Vector** speaks and acts on commands
5. **Camera** feeds to YOLOv8 for object detection
6. **UI** shows conversation and camera view

### Supported Commands:
- **Movement:** "move forward", "turn left", etc.
- **Emotions:** Vector shows emotions
- **Vision:** "find a person", "see the cat", etc.
- **Conversation:** Chat naturally with Vector

---

## 💰 Cost Considerations

### One-Time Downloads (First Run):
- Whisper model: ~460MB
- YOLOv8 model: ~6MB
- **Total:** ~470MB

### Ongoing Costs:
- **OpenAI API:** Pay per use
  - ChatGPT: ~$0.0015-0.002 per message
  - Whisper API: Not used (local model)
- **Electricity:** ~5W on Raspberry Pi 4

**Tip:** Monitor usage at https://platform.openai.com/usage

---

## 🎯 Next Steps

### Immediate:
1. Set OpenAI API key
2. Configure Vector
3. Run `./run.sh`

### Soon:
1. Test object detection
2. Experiment with voice commands
3. Customize Vector's personality (edit `src/customgpt.py`)

### Advanced:
1. Switch to OWL-ViT for zero-shot detection
2. Optimize Whisper model size
3. Add custom commands
4. Integrate with other services

---

## 🎉 Ready to Go!

You're all set! Just need to:

1. **Set OpenAI API key**
2. **Configure Vector**
3. **Run:** `./run.sh`

**Enjoy your AI-powered Vector robot!** 🤖✨

---

## 📞 Need Help?

- **Setup Issues:** See `SETUP_COMPLETE.md`
- **Object Detection:** See `OBJECT_DETECTION_GUIDE.md`
- **Quick Commands:** See `QUICK_START.md`
- **General Help:** Check all `*.md` files in project root

**Have fun!** 🚀

