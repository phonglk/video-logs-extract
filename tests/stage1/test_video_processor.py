"""Tests for the VideoProcessor class in stage1."""

import pytest
from pathlib import Path
import json
import cv2
import numpy as np
from src.stage1.video_processor import VideoProcessor
from src.stage1.person_detector import PersonDetector
from src.stage1.utils import ProgressManager

def test_video_processor_init(test_config):
    """Test VideoProcessor initialization."""
    processor = VideoProcessor(test_config)
    assert processor is not None
    assert processor.config == test_config
    assert isinstance(processor.detector, PersonDetector)

def test_process_video(test_video, test_output_dir, test_config, monkeypatch):
    """Test video processing with a test video."""
    # Mock the person detector to always return a detection
    class MockDetector:
        def process_batch(self, frames):
            return [(1, [{'bbox': [0, 0, 100, 200], 'confidence': 0.9}]) 
                   for _ in frames]
    
    processor = VideoProcessor(test_config)
    monkeypatch.setattr(processor, 'detector', MockDetector())
    
    # Process the test video
    progress = ProgressManager()
    processor.process_video(str(test_video), progress)
    
    # Check that output files were created
    output_files = list(Path(test_config['output']['directory']).glob('**/*.jpg'))
    assert len(output_files) > 0
    
    # Check that metadata files were created
    metadata_files = list(Path(test_config['output']['directory']).glob('**/*.json'))
    assert len(metadata_files) > 0
    
    # Verify metadata content
    with open(metadata_files[0]) as f:
        metadata = json.load(f)
        assert 'detections' in metadata
        assert isinstance(metadata['detections'], list)
        assert len(metadata['detections']) > 0

def test_prepare_frame_batch(test_config):
    """Test frame batch preparation."""
    processor = VideoProcessor(test_config)
    
    # Create test frames
    frames = [
        np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(4)
    ]
    
    # Process frames
    processed = processor.prepare_frame_batch(frames)
    
    # Check dimensions
    target_height = test_config['detection']['resize_height']
    target_width = test_config['detection']['resize_width']
    
    assert len(processed) == len(frames)
    assert processed[0].shape[0] == target_height
    assert processed[0].shape[1] == target_width
    assert processed[0].shape[2] == 3

def test_process_batch(test_config):
    """Test batch processing."""
    processor = VideoProcessor(test_config)
    
    # Create test frames
    frames = [
        np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(4)
    ]
    
    # Create timestamps
    from datetime import datetime, timedelta
    base_time = datetime.now()
    timestamps = [base_time + timedelta(seconds=i) for i in range(len(frames))]
    
    # Create progress manager
    progress = ProgressManager()
    
    # Process batch
    processor._process_batch(frames, timestamps, progress, 1.0)
    
    # Check progress was updated
    assert progress.processed_frames == len(frames)

def test_error_handling(test_config, temp_dir):
    """Test error handling for invalid inputs."""
    processor = VideoProcessor(test_config)
    
    # Test with non-existent video
    progress = ProgressManager()
    with pytest.raises(Exception) as exc_info:
        processor.process_video(
            str(temp_dir / 'nonexistent.mp4'),
            progress
        )
    assert "Failed to open video" in str(exc_info.value)

def test_performance_metrics(test_video, test_output_dir, test_config):
    """Test that performance metrics are tracked."""
    processor = VideoProcessor(test_config)
    
    # Process video
    progress = ProgressManager()
    processor.process_video(str(test_video), progress)
    
    # Check metrics
    assert len(processor.perf_metrics['read_time']) > 0
    assert len(processor.perf_metrics['resize_time']) > 0
    assert len(processor.perf_metrics['detect_time']) > 0
    assert len(processor.perf_metrics['save_time']) > 0 