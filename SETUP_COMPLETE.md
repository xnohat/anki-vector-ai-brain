# Complete Setup Guide for Vector Advanced AI

## ✅ Dependencies Installed

All required dependencies have been installed:

### Object Detection
- ✅ `ultralytics` (YOLOv8) - Currently active
- ✅ `torch`, `torchvision`, `transformers` - For OWL-ViT alternative
- ✅ `opencv-python`, `Pillow`, `numpy` - Image processing

### UI
- ✅ `customtkinter` - Modern UI framework

### Speech/Audio
- ✅ `openai-whisper` - Speech-to-text
- ✅ `sounddevice` - Audio input
- ✅ `scipy` - Audio processing

### AI Integration
- ✅ `openai` - ChatGPT API

### Vector Robot
- ✅ `kingardor-vector` (anki_vector) - Vector SDK

---

## 🔑 Required API Keys

### OpenAI API Key (Required)

The app uses OpenAI for:
1. **ChatGPT** - AI conversation
2. **Whisper** - Speech-to-text (uses local model, but OpenAI package needed)

**Set your API key:**

```bash
# Method 1: Environment variable (recommended)
export OPENAI_API_KEY='your-api-key-here'

# To make it permanent, add to ~/.bashrc:
echo "export OPENAI_API_KEY='your-api-key-here'" >> ~/.bashrc
source ~/.bashrc
```

```bash
# Method 2: Create .env file (if supported)
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

**Get your API key:**
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Copy and save it securely

---

## 🤖 Vector Robot Configuration

### Install System Dependencies

Install required system libraries:

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-tk
```

### Configure Vector Connection

Run the Vector SDK configuration:

```bash
source .venv/bin/activate
python -m anki_vector.configure
```

This will:
1. Find your Vector robot on the network
2. Download authentication credentials
3. Save configuration to `~/.anki_vector/sdk_config.ini`

**Requirements:**
- Vector robot must be on the same WiFi network
- Vector must be running wire-pod (local server)
- wire-pod documentation: https://github.com/kercre123/wire-pod

---

## 🚀 Running the App

### Option 1: Using the launcher script (Easiest)
```bash
./run.sh
```

### Option 2: Manual activation
```bash
source .venv/bin/activate
python app.py
```

### Option 3: Direct execution
```bash
.venv/bin/python app.py
```

---

## 🧪 Test Components Individually

### Test Object Detection Only
```bash
source .venv/bin/activate
python scripts/test_detection.py
```

### Test Whisper STT
```bash
source .venv/bin/activate
python scripts/whisper-test.py
```

### Test OpenAI Connection
```bash
source .venv/bin/activate
python scripts/openai-test.py
```

---

## 🔧 Common Issues & Solutions

### Issue: "ModuleNotFoundError"
**Solution:** Make sure virtual environment is activated
```bash
source .venv/bin/activate
```

### Issue: "OpenAI API key not found"
**Solution:** Set the OPENAI_API_KEY environment variable
```bash
export OPENAI_API_KEY='your-key-here'
```

### Issue: "Vector robot not found"
**Solution:** 
1. Check Vector is on same WiFi
2. Run configuration: `python -m anki_vector.configure`
3. Check wire-pod is running

### Issue: "Whisper model downloading slowly"
**Solution:** First run downloads the model (~460MB for 'small'). Be patient!
Models are cached at `~/.cache/whisper/`

### Issue: "Audio input not working"
**Solution:** 
1. Check microphone permissions
2. Test with: `python -m sounddevice`
3. May need to configure ALSA on Raspberry Pi

### Issue: "UI not showing / Tkinter error"
**Solution:**
1. Make sure you're running with display: `export DISPLAY=:0`
2. Install: `sudo apt-get install python3-tk`
3. For headless: The UI requires a display

### Issue: "Out of memory"
**Solution:**
1. Increase swap: See MIGRATION_SUMMARY.md
2. Use YOLOv8 instead of OWL-ViT (already configured)
3. Use smaller Whisper model: Edit `src/whisperstt.py` change 'small' to 'tiny' or 'base'

---

## 📁 Project Structure

```
vector-advanced-ai/
├── app.py                      # Main application
├── run.sh                      # Launch script
├── requirements.txt            # Python dependencies
│
├── src/                        # Source code
│   ├── owl.py                 # Object detection (YOLOv8)
│   ├── owl_yolo.py            # YOLOv8 implementation (backup)
│   ├── customgpt.py           # ChatGPT integration
│   ├── speechstream.py        # Audio streaming
│   ├── whisperstt.py          # Speech-to-text
│   ├── ui.py                  # User interface
│   └── vectorbot.py           # Vector robot control
│
├── scripts/                    # Utility scripts
│   ├── test_detection.py      # Test object detection
│   ├── install_rpi.sh         # Installation script
│   └── [other test scripts]
│
├── docs/                       # Documentation
│   ├── QUICK_START.md
│   ├── OBJECT_DETECTION_GUIDE.md
│   ├── MIGRATION_SUMMARY.md
│   ├── WHICH_ONE_TO_USE.md
│   └── SETUP_COMPLETE.md      # This file
│
└── vector-python-sdk/         # Vector SDK (submodule)
```

---

## 🎯 Quick Checklist

Before running the app, make sure:

- [ ] Virtual environment created and activated
- [ ] All Python packages installed (`pip install -r requirements.txt`)
- [ ] Vector SDK installed (`pip install -e vector-python-sdk/`)
- [ ] OpenAI API key set (environment variable)
- [ ] Vector robot configured (`python -m anki_vector.configure`)
- [ ] Vector on same WiFi network
- [ ] wire-pod server running
- [ ] Microphone connected and working
- [ ] Display available (for UI)

---

## 💡 Optimization Tips

### For Faster Performance:
1. **Object Detection:** Already using YOLOv8 (optimal)
2. **Whisper:** Use 'tiny' or 'base' model instead of 'small'
3. **Memory:** Close other applications
4. **CPU:** Consider overclocking RPi (if experienced)

### For Lower Resource Usage:
1. **Whisper:** Use 'tiny' model (edit `src/whisperstt.py` line 8)
2. **Object Detection:** Already optimal with YOLOv8
3. **Swap:** Add swap space if needed (see MIGRATION_SUMMARY.md)

---

## 📞 Getting Help

1. **Quick Reference:** See `QUICK_START.md`
2. **Object Detection:** See `OBJECT_DETECTION_GUIDE.md`
3. **Migration Info:** See `MIGRATION_SUMMARY.md`
4. **Which Implementation:** See `WHICH_ONE_TO_USE.md`

---

## 🎉 You're All Set!

Run the app with:
```bash
./run.sh
```

Or:
```bash
source .venv/bin/activate
python app.py
```

Enjoy your AI-powered Vector robot! 🤖✨

