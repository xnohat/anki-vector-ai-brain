"""
Alternative YOLO-based object detector for Raspberry Pi.
This is faster than OWL-ViT but uses pre-trained COCO classes instead of zero-shot detection.

To use this instead of the transformers-based owl.py:
1. Install ultralytics: pip install ultralytics
2. Rename owl.py to owl_transformers.py
3. Rename this file to owl.py
"""

from typing import Tuple, Dict
import PIL.Image
import numpy as np
from ultralytics import YOLO
from PIL import ImageDraw, ImageFont


class HootHoot:
    def __init__(
            self, 
            model: str = "yolov8n.pt",  # yolov8n is the fastest/smallest model
            device: str = "cpu"
    ) -> None:
        """
        Initialize YOLOv8 object detector for Raspberry Pi.
        Much faster than OWL-ViT but limited to COCO dataset classes.
        
        Args:
            model: YOLOv8 model name (yolov8n.pt, yolov8s.pt, etc.)
            device: Device to run on ("cpu" for Raspberry Pi)
        """
        print(f"Loading {model} model on {device}...")
        self.model = YOLO(model)
        self.device = device
        
        # COCO dataset class names
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        self.prompt = '[]'
        self.filter_classes = None
        print("Model loaded successfully!")
        print(f"Available classes: {', '.join(self.coco_classes)}")
    
    def _parse_prompt(self, prompt: str) -> list:
        """
        Parse prompt string to extract labels.
        Matches them against available COCO classes.
        """
        import re
        
        # Remove brackets and split by comma
        prompt = prompt.strip()
        if prompt.startswith('[') and prompt.endswith(']'):
            prompt = prompt[1:-1]
        
        if not prompt:
            return None  # Return all classes
        
        # Split by comma and clean up
        labels = [label.strip().lower() for label in prompt.split(',')]
        # Remove 'a' and 'an' prefixes
        labels = [re.sub(r'^(a|an)\s+', '', label) for label in labels]
        
        # Match against COCO classes
        matched_classes = []
        for label in labels:
            for i, coco_class in enumerate(self.coco_classes):
                if label in coco_class or coco_class in label:
                    if i not in matched_classes:
                        matched_classes.append(i)
        
        return matched_classes if matched_classes else None
    
    def _draw_predictions(
        self, 
        image: PIL.Image.Image, 
        results
    ) -> PIL.Image.Image:
        """
        Draw bounding boxes and labels on the image using YOLO results.
        """
        draw = ImageDraw.Draw(image)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        colors = [
            'red', 'green', 'blue', 'yellow', 'purple', 
            'orange', 'pink', 'cyan', 'magenta', 'lime'
        ]
        
        for i, result in enumerate(results):
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = self.coco_classes[class_id]
                
                color = colors[i % len(colors)]
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw label with background
                text = f"{label}: {confidence:.2f}"
                bbox = draw.textbbox((x1, y1), text, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x1, y1), text, fill='white', font=font)
        
        return image

    def predict(
            self, 
            image: PIL.Image.Image, 
            prompt: str,
            threshold: float = 0.1
        ) -> Tuple[Dict, PIL.Image.Image]:
        """
        Perform object detection on the image using YOLOv8.
        
        Args:
            image: PIL Image to detect objects in
            prompt: String with object labels (e.g., "[person, car]")
                   Only detects objects that match COCO classes
            threshold: Confidence threshold for detections (default: 0.1)
            
        Returns:
            Tuple of (output dict, annotated image)
        """
        # Parse labels from prompt
        if self.prompt != prompt:
            self.prompt = prompt
            self.filter_classes = self._parse_prompt(prompt)
        
        # Run inference
        results = self.model.predict(
            image, 
            conf=threshold, 
            device=self.device,
            verbose=False
        )
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                
                # Filter by requested classes if specified
                if self.filter_classes is not None and class_id not in self.filter_classes:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                label = self.coco_classes[class_id]
                
                detections.append({
                    "label": label,
                    "score": confidence,
                    "box": [float(x1), float(y1), float(x2), float(y2)]
                })
        
        output = {
            "detections": detections,
            "labels": [self.coco_classes[i] for i in self.filter_classes] if self.filter_classes else self.coco_classes,
            "count": len(detections)
        }
        
        # Draw predictions on image
        annotated_image = image.copy()
        if detections:
            annotated_image = self._draw_predictions(annotated_image, results)
        
        return output, annotated_image

