from pathlib import Path
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO
import torch
import sys
import os
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import time

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
        
        # Get debug mode from config
        self.debug = config['processing'].get('log_level', 'INFO') == 'DEBUG'
        
        # Log detection parameters
        self.logger.info(f"Initializing detector with confidence={self.confidence}, "
                        f"target_persons={self.target_persons}")
        
        # Load model with complete output suppression
        with suppress_output():
            # Initialize YOLO model without callbacks
            model_path = config['detection'].get('model', 'yolov8n.pt')
            self.logger.info(f"Loading YOLO model from {model_path}")
            
            # Improved model loading with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.model = YOLO(model_path, verbose=False)
                    break
                except Exception as e:
                    self.logger.error(f"Error loading model (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)  # Wait before retry
            
            # Configure model settings for Apple Silicon
            if torch.backends.mps.is_available() and config['processing'].get('use_gpu', True):
                self.logger.info("Using MPS (Metal Performance Shaders) backend")
                self.device = torch.device("mps")
                # Move model to MPS device
                self.model.to(self.device)
            elif torch.cuda.is_available() and config['processing'].get('use_gpu', True):
                self.logger.info("Using CUDA backend")
                self.device = torch.device("cuda:0")
                self.model.to(self.device)
            else:
                self.logger.warning("GPU acceleration not available, falling back to CPU")
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
            
        batch_size = min(4, self.config['processing'].get('batch_size', 8))
        self.logger.info(f"Warming up model with batch size {batch_size}")
        dummy_batch = [np.zeros((416, 416, 3), dtype=np.uint8) for _ in range(batch_size)]
        
        # Single warmup iteration is sufficient for MPS
        with suppress_output():
            self.model(dummy_batch, verbose=False)
        self.logger.info("Model warmup completed")

    def process_batch(self, frames: List[np.ndarray]) -> List[Optional[Tuple[int, List[Dict[str, Any]]]]]:
        """Process a batch of frames and return detection results."""
        if not frames:
            return []
        
        start_time = time.time()
        
        # Print debug info about input frames
        if self.debug:
            for i, frame in enumerate(frames[:3]):  # Log first 3 frames
                self.logger.debug(f"Frame {i} shape: {frame.shape}, dtype: {frame.dtype}, "
                               f"min: {frame.min()}, max: {frame.max()}")
        
        # Run inference with complete output suppression
        with suppress_output():
            results = self.model(frames, stream=True, verbose=False)
        
        # Process results
        batch_results = []
        all_detections_count = 0
        frames_with_detections = 0
        saved_frames = 0
                
        for i, result in enumerate(results):
            # Debug info for raw detection results
            if self.debug and i < 3:  # Log first 3 frames
                boxes = result.boxes
                self.logger.debug(f"Frame {i}: {len(boxes)} total detections before filtering")
            
            # Filter for person class (class 0 in COCO) and confidence threshold
            person_dets = result.boxes[
                (result.boxes.cls == 0) & 
                (result.boxes.conf >= self.confidence)
            ]
            num_persons = len(person_dets)
            all_detections_count += num_persons
            
            if num_persons > 0:
                frames_with_detections += 1
            
            # Log all detections for debugging
            if self.debug and i < 5:  # Log first 5 frames
                self.logger.debug(f"Frame {i}: {num_persons} persons detected")
                for j, box in enumerate(person_dets):
                    self.logger.debug(f"  Person {j}: confidence={float(box.conf[0]):.3f}, "
                                  f"bbox={box.xyxy[0].tolist()}")
            
            # Save if we have at least the target number of persons detected
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
                saved_frames += 1
            else:
                batch_results.append(None)
        
        elapsed = time.time() - start_time
        fps = len(frames) / elapsed if elapsed > 0 else 0
        
        self.logger.info(f"Processed {len(frames)} frames in {elapsed:.3f}s ({fps:.1f} FPS)")
        self.logger.info(f"Found {all_detections_count} total persons in {frames_with_detections}/{len(frames)} frames")
        self.logger.info(f"Saving {saved_frames}/{len(frames)} frames that meet criteria (≥{self.target_persons} persons)")
                
        return batch_results

    def __del__(self):
        """Cleanup memory if needed."""
        if hasattr(self, 'model'):
            del self.model 