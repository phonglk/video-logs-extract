"""Test configuration and fixtures for stage1 tests."""

import pytest
import os
import tempfile
from pathlib import Path
import cv2
import numpy as np
import yaml

@pytest.fixture
def test_config(test_output_dir):
    """Create a test configuration."""
    return {
        'detection': {
            'model': 'yolov8n.pt',
            'confidence_threshold': 0.5,
            'person_count': 1,
            'skip_frames': 2,
            'resize_width': 416,
            'resize_height': 416,
            'maintain_aspect_ratio': True,
            'iou_threshold': 0.5,
            'max_detections': 20,
            'confidence': 0.5,
            'target_person_count': 1,
            'device': 'cpu'
        },
        'output': {
            'format': '{timestamp}.jpg',
            'quality': 90,
            'min_interval_seconds': 1,
            'include_metadata': True,
            'metadata_format': '{timestamp}.json',
            'stage1_raw': str(test_output_dir / 'stage1_raw'),
            'stage2_processed': str(test_output_dir / 'stage2_processed'),
            'directory': str(test_output_dir)  # Add the output directory
        },
        'processing': {
            'batch_size': 4,
            'use_gpu': False,
            'half_precision': False,
            'num_threads': 1,
            'buffer_size': 8,
            'log_level': 'DEBUG',
            'frame_skip': 2,
            'frame_queue_size': 16,
            'result_queue_size': 8,
            'prefetch_batches': 2
        }
    }

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def test_video(temp_dir):
    """Create a test video file."""
    video_path = temp_dir / 'test.mp4'
    
    # Create a simple test video
    width, height = 640, 480
    fps = 30.0
    duration = 2  # seconds
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    try:
        # Generate frames
        for i in range(int(fps * duration)):
            # Create a frame with a rectangle that could be detected as a person
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Draw a person-like rectangle
            cv2.rectangle(frame, (width//4, height//4), (3*width//4, 3*height//4), (255, 255, 255), -1)
            out.write(frame)
    finally:
        out.release()
    
    yield video_path
    
@pytest.fixture
def test_input_dir(temp_dir, test_video):
    """Create a test input directory with test videos."""
    input_dir = temp_dir / 'input'
    input_dir.mkdir()
    
    # Copy test video with timestamp-based name
    from datetime import datetime
    timestamp = int(datetime.now().timestamp())
    video_name = f'00M00S_{timestamp}.mp4'
    os.symlink(test_video, input_dir / video_name)
    
    return input_dir

@pytest.fixture
def test_output_dir(temp_dir):
    """Create and return a test output directory."""
    output_dir = temp_dir / 'output'
    output_dir.mkdir()
    
    # Create stage directories
    (output_dir / 'stage1_raw').mkdir()
    (output_dir / 'stage2_processed').mkdir()
    
    return output_dir

@pytest.fixture
def test_config_file(temp_dir, test_config):
    """Create a test configuration file."""
    config_file = temp_dir / 'test_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(test_config, f)
    return config_file 