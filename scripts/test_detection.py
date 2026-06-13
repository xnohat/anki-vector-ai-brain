#!/usr/bin/env python3
"""
Test script for object detection on Raspberry Pi.
Tests the new Raspberry Pi-compatible object detection implementation.
"""

import sys
sys.path.insert(0, '../src')
sys.path.insert(0, 'src')

import time
import PIL.Image
from pathlib import Path

def create_test_image():
    """Create a simple test image if no image is available."""
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a test image with some shapes
    img = Image.new('RGB', (640, 480), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes
    draw.rectangle([100, 100, 200, 200], fill='red', outline='black', width=3)
    draw.ellipse([300, 150, 400, 250], fill='green', outline='black', width=3)
    draw.polygon([(500, 350), (450, 450), (550, 450)], fill='blue', outline='black', width=3)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    draw.text((150, 300), "TEST IMAGE", fill='black', font=font)
    
    return img

def test_detection():
    """Test the object detection implementation."""
    print("=" * 60)
    print("Object Detection Test for Raspberry Pi")
    print("=" * 60)
    print()
    
    # Import the detector
    try:
        import owl
        print("✓ Successfully imported owl module")
    except ImportError as e:
        print(f"✗ Failed to import owl module: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        return False
    
    # Initialize detector
    print("\nInitializing detector...")
    try:
        detector = owl.HootHoot()
        print("✓ Detector initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize detector: {e}")
        return False
    
    # Load or create test image
    print("\nLoading test image...")
    test_image_path = Path("resources/test_image.jpg")
    
    if test_image_path.exists():
        image = PIL.Image.open(test_image_path)
        print(f"✓ Loaded test image from {test_image_path}")
    else:
        print("! No test image found, creating a simple test image...")
        image = create_test_image()
        print("✓ Created test image")
    
    print(f"  Image size: {image.size}")
    
    # Run detection with different prompts
    test_prompts = [
        "[a person, a car]",
        "[person, bottle, chair]",
        "[cat, dog, bird]"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'=' * 60}")
        print(f"Test {i}/3: Detection with prompt: {prompt}")
        print('=' * 60)
        
        try:
            start_time = time.time()
            output, annotated_image = detector.predict(
                image=image,
                prompt=prompt,
                threshold=0.1
            )
            inference_time = time.time() - start_time
            
            print(f"✓ Detection completed in {inference_time:.2f} seconds")
            print(f"  Found {output['count']} objects")
            
            if output['detections']:
                print("\n  Detections:")
                for detection in output['detections']:
                    label = detection['label']
                    score = detection['score']
                    print(f"    - {label}: {score:.2%} confidence")
            else:
                print("  No objects detected (try adjusting threshold)")
            
            # Save annotated image
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"test_result_{i}.jpg"
            annotated_image.save(output_path)
            print(f"\n  Saved annotated image to: {output_path}")
            
        except Exception as e:
            print(f"✗ Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 60)
    print("All tests completed successfully! ✓")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check the 'output/' folder for annotated images")
    print("2. Try with your own images")
    print("3. Adjust the threshold parameter for better results")
    print("4. See OBJECT_DETECTION_GUIDE.md for optimization tips")
    print()
    
    return True

def main():
    """Main function."""
    success = test_detection()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

