"""
Stage 1: Video Frame Extraction with Person Detection

This package handles the initial processing of video files to extract frames
containing people, using YOLOv8 for detection and providing detailed metadata.
"""

# Import optimized components from base module
from ..optimized_processor import OptimizedFrameBuffer, OptimizedVideoProcessor as BaseOptimizedProcessor, parallel_resize

# Import stage1-specific components
from .person_detector import PersonDetector
from .utils import (
    get_video_info,
    FrameManager,
    ProgressManager,
    BatchProgressManager,
    resize_frame
)

# Import the specialized video processor for stage1
from .optimized_video_processor import OptimizedVideoProcessor, create_from_config

__all__ = [
    'OptimizedVideoProcessor',
    'PersonDetector',
    'OptimizedFrameBuffer',  # Use the optimized frame buffer from base
    'create_from_config',
    'get_video_info',
    'FrameManager',
    'ProgressManager',
    'BatchProgressManager',
    'resize_frame',
    'parallel_resize'
] 