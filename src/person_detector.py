from pathlib import Path
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO
import torch
import sys
import os
from contextlib import contextmanager, redirect_stdout, redirect_stderr

@contextmanager
def suppress_output():
    """Context manager to completely suppress all output including YOLO's progress bars."""
    # Create a null device to redirect output
    with open(os.devnull, 'w') as devnull:
        # Redirect both stdout and stderr
        with redirect_stdout(devnull), redirect_stderr(devnull):
            try:
                yield
            finally:
                pass

class PersonDetector:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the person detector with configuration."""
        self.config = config
        self.confidence = config['detection'].get('confidence_threshold', 0.5)
        self.target_persons = config['detection'].get('person_count', 2)
        self.use_half = config['processing'].get('half_precision', True)
        self.logger = logging.getLogger(__name__)
        
        # Load model with complete output suppression
        with suppress_output():
            # Initialize YOLO model without callbacks
            self.model = YOLO(
                config['detection'].get('model', 'yolov8n.pt'),
                verbose=False
            )
            
            # Configure model settings for Apple Silicon
            if torch.backends.mps.is_available() and config['processing'].get('use_gpu', True):
                self.logger.info("Using MPS (Metal Performance Shaders) backend")
                self.device = torch.device("mps")
                # Move model to MPS device
                self.model.to(self.device)
            else:
                self.logger.warning("MPS not available, falling back to CPU")
                self.device = torch.device("cpu")
                self.model.to(self.device)
            
            # Set model parameters
            self.model.conf = self.confidence
            self.model.iou = config['detection'].get('iou_threshold', 0.5)
            self.model.max_det = config['detection'].get('max_detections', 20)
            
            # Warm up the model
            self.warmup()

    def warmup(self):
        """Warm up the model with a dummy batch."""
        if not hasattr(self, 'model'):
            return
            
        batch_size = self.config['processing'].get('batch_size', 8)  # Smaller batch size for M1
        dummy_batch = [np.zeros((416, 416, 3), dtype=np.uint8) for _ in range(batch_size)]
        
        # Single warmup iteration is sufficient for MPS
        with suppress_output():
            self.model(dummy_batch, verbose=False)

    def process_batch(self, frames: List[np.ndarray]) -> List[Optional[Tuple[int, List[Dict[str, Any]]]]]:
        """Process a batch of frames and return detection results."""
        if not frames:
            return []
            
        # Run inference with complete output suppression
        with suppress_output():
            results = self.model(frames, stream=True, verbose=False)
            
        batch_results = []
        for result in results:
            # Filter for person class (class 0 in COCO) and confidence threshold
            person_dets = result.boxes[
                (result.boxes.cls == 0) & 
                (result.boxes.conf >= self.confidence)
            ]
            num_persons = len(person_dets)
            
            # Save if we have at least 1 person detected
            if num_persons >= self.target_persons:
                # Convert detections to list of dicts
                detections = []
                for box in person_dets:
                    det = {
                        'bbox': box.xyxy[0].tolist(),
                        'confidence': float(box.conf[0]),
                        'class': 'person'
                    }
                    detections.append(det)
                batch_results.append((num_persons, detections))
            else:
                batch_results.append(None)
                
        return batch_results

    def __del__(self):
        """Cleanup memory if needed."""
        if hasattr(self, 'model'):
            del self.model 