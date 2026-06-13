#!/bin/bash
# Installation script for Vector Advanced AI on Raspberry Pi
# Replaces nanoowl with Raspberry Pi-compatible alternatives

set -e  # Exit on error

echo "=========================================="
echo "Vector Advanced AI - Raspberry Pi Setup"
echo "=========================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null && ! grep -q "BCM" /proc/cpuinfo 2>/dev/null; then
    echo -e "${YELLOW}Warning: This doesn't appear to be a Raspberry Pi${NC}"
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Step 1: Updating system packages..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev libjpeg-dev zlib1g-dev libopenblas-dev

echo
echo "Step 2: Setting up Python virtual environment (recommended)..."
echo "Create virtual environment? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
        echo -e "${GREEN}✓ Virtual environment created${NC}"
    else
        echo -e "${YELLOW}Virtual environment already exists${NC}"
    fi
    source .venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
fi

echo
echo "Step 3: Choose object detection method:"
echo "  1) OWL-ViT with Transformers (Zero-shot, slower, ~500MB)"
echo "  2) YOLOv8 (Faster, 80 classes only, ~6MB)"
echo "  3) Install both (for comparison)"
echo
read -p "Enter choice (1/2/3): " choice

case $choice in
    1)
        echo
        echo "Installing OWL-ViT with Transformers..."
        pip3 install torch torchvision transformers pillow numpy
        echo -e "${GREEN}✓ OWL-ViT installed${NC}"
        echo
        echo "Current implementation (src/owl.py) is already configured for OWL-ViT"
        ;;
    2)
        echo
        echo "Installing YOLOv8..."
        pip3 install ultralytics pillow numpy
        echo -e "${GREEN}✓ YOLOv8 installed${NC}"
        echo
        echo "Switching to YOLO implementation..."
        if [ -f "src/owl.py" ]; then
            mv src/owl.py src/owl_transformers.py.bak
            echo "  Backed up OWL-ViT implementation to src/owl_transformers.py.bak"
        fi
        if [ -f "src/owl_yolo.py" ]; then
            cp src/owl_yolo.py src/owl.py
            echo -e "${GREEN}✓ Switched to YOLO implementation${NC}"
        else
            echo -e "${RED}Error: src/owl_yolo.py not found${NC}"
            exit 1
        fi
        ;;
    3)
        echo
        echo "Installing both implementations..."
        pip3 install torch torchvision transformers ultralytics pillow numpy
        echo -e "${GREEN}✓ Both implementations installed${NC}"
        echo
        echo "Default is OWL-ViT. To switch to YOLO, run:"
        echo "  mv src/owl.py src/owl_transformers.py"
        echo "  cp src/owl_yolo.py src/owl.py"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo
echo "Step 4: Installing Vector Python SDK..."
if [ -d "vector-python-sdk" ]; then
    pip3 install -e vector-python-sdk/
    echo -e "${GREEN}✓ Vector SDK installed${NC}"
else
    echo -e "${YELLOW}Warning: vector-python-sdk directory not found${NC}"
    echo "Clone it from: https://github.com/kingardor/vector-python-sdk"
fi

echo
echo "=========================================="
echo -e "${GREEN}Installation Complete!${NC}"
echo "=========================================="
echo
echo "Next steps:"
echo "  1. Test the installation:"
echo "     python3 scripts/test_detection.py"
echo
echo "  2. Read the guide:"
echo "     cat OBJECT_DETECTION_GUIDE.md"
echo
echo "  3. Run the main application:"
echo "     python3 app.py"
echo
echo "Tips:"
echo "  - First run will download models (~500MB for OWL-ViT or ~6MB for YOLO)"
echo "  - See OBJECT_DETECTION_GUIDE.md for performance optimization"
echo "  - For YOLO class list, run: python3 -c 'from ultralytics import YOLO; print(YOLO(\"yolov8n.pt\").names)'"
echo

