import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
import cv2
import numpy as np
import time
import sys
from typing import Optional, Dict
import click
import os
import shutil

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Configure logging to use a file instead of stdout
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='video_extract.log'
    )

class BatchProgressManager:
    """Manages progress for a batch of videos."""
    def __init__(self, total_videos: int):
        self.total_videos = total_videos
        self.completed_videos = 0
        self.start_time = time.time()
        
    def update_progress(self, video_progress: float) -> None:
        """Update overall batch progress."""
        total_progress = (self.completed_videos + video_progress) / self.total_videos
        self._display_batch_progress(total_progress)
        
    def video_complete(self) -> None:
        """Mark a video as complete."""
        self.completed_videos += 1
        
    def _display_batch_progress(self, progress: float) -> None:
        """Display batch progress information."""
        elapsed_time = time.time() - self.start_time
        if progress > 0:
            estimated_total_time = elapsed_time / progress
            remaining_time = estimated_total_time - elapsed_time
        else:
            remaining_time = 0
            
        stats = (
            f"\nBatch Progress: {progress*100:.1f}% "
            f"| Videos: {self.completed_videos}/{self.total_videos} "
            f"| Elapsed: {timedelta(seconds=int(elapsed_time))} "
            f"| ETA: {timedelta(seconds=int(remaining_time))}"
        )
        
        # Print batch progress on a separate line
        sys.stdout.write('\n\033[1A' + stats + '\n')

class ProgressManager:
    """Manages progress for a single video process."""
    
    def __init__(self, batch_manager=None):
        """Initialize progress manager."""
        self.batch_manager = batch_manager
        self.processed_frames = 0
        self.saved_frames = 0
        self.skipped_frames = 0
        self.total_frames = 0
        self.current_fps = 0
        self.video_fps = 0
        self.video_duration = 0
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.terminal_width = shutil.get_terminal_size().columns
        self.video_filename = ""
        self.processing_date = datetime.now()
        
    def set_total(self, total: int) -> None:
        """Set total frames to process."""
        self.total_frames = max(1, total)
        
    def set_video_info(self, fps: float, duration: float, filename: str = "") -> None:
        """Set video information."""
        self.video_fps = fps
        self.video_duration = duration
        self.video_filename = filename
        self.processing_date = datetime.now()
        
        # Try to extract timestamp from video filename if available
        self.video_timestamp = None
        try:
            # Assuming filename format like "00M44S_1728867644.mp4"
            basename = os.path.basename(filename)
            parts = basename.split('_')
            if len(parts) >= 2:
                ts_part = parts[1].split('.')[0]
                self.video_timestamp = int(ts_part)
        except:
            pass
        
    def update_batch_stats(self, fps: float) -> None:
        """Update batch processing statistics."""
        self.current_fps = fps
        
    def update(self, processed: int = 0, saved: int = 0, skipped: int = 0, batch_size: int = 0) -> None:
        """Update progress counters and display."""
        self.processed_frames += processed
        self.saved_frames += saved
        self.skipped_frames += skipped
        
        # Don't update too frequently to avoid screen flicker
        current_time = time.time()
        if current_time - self.last_update_time >= 0.1 or processed >= batch_size:
            self._display_progress()
            self.last_update_time = current_time
            
    def complete(self) -> None:
        """Mark processing as complete and show final statistics."""
        self._display_progress(force=True)
        print()  # Add newline after progress
        
    def _display_progress(self, force: bool = False) -> None:
        """Display progress information."""
        # Calculate progress percentage
        progress = (self.processed_frames + self.skipped_frames) / self.total_frames if self.total_frames > 0 else 0
        
        # Calculate time statistics
        elapsed_time = time.time() - self.start_time
        if progress > 0:
            estimated_total_time = elapsed_time / progress
            remaining_time = estimated_total_time - elapsed_time
        else:
            remaining_time = 0
            
        # Create progress bar
        bar_width = min(50, self.terminal_width - 30)
        filled_width = int(bar_width * progress)
        bar = '=' * filled_width + '>' + ' ' * (bar_width - filled_width)
        
        # Format date and filename info
        # Use video timestamp if available
        date_str = ""
        if hasattr(self, 'video_timestamp') and self.video_timestamp:
            # If we have a timestamp from the video filename, use that
            try:
                video_date = datetime.fromtimestamp(self.video_timestamp)
                date_str = video_date.strftime("%Y-%m-%d %H:%M:%S")
            except:
                # Fall back to processing date if timestamp conversion fails
                date_str = self.processing_date.strftime("%Y-%m-%d %H:%M:%S")
        else:
            # Use processing start time if no video timestamp
            date_str = self.processing_date.strftime("%Y-%m-%d %H:%M:%S")
            
        filename = os.path.basename(self.video_filename) if self.video_filename else "Unknown"
        
        # Format a compact filename if it's too long
        if len(filename) > 20:
            filename = filename[:17] + "..."
            
        # Format statistics
        stats = (
            f"[{bar}] {progress*100:5.1f}% "
            f"| Processed: {self.processed_frames}/{self.total_frames} "
            f"| Saved: {self.saved_frames} "
            f"| FPS: {self.current_fps:.1f} "
            f"| ETA: {timedelta(seconds=int(remaining_time))}"
        )
        
        # Create a single-line progress display
        # Include the date and filename in the same line as the progress stats
        info_line = f"{date_str} | {filename} | {stats}"
        
        # Use carriage return to ensure we always overwrite the current line
        # without adding newlines that could cause duplication
        sys.stdout.write('\r' + ' ' * self.terminal_width)  # Clear the line
        sys.stdout.write('\r' + info_line[:self.terminal_width-1])
        sys.stdout.flush()
        
        # Update batch progress if available
        if self.batch_manager:
            self.batch_manager.update_progress(progress)

def get_video_info(video_path: Path) -> dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(str(video_path))
    info = {
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    }
    cap.release()
    return info

class FrameManager:
    def __init__(self, output_dir: Path, min_interval: int = 5):
        """Initialize frame manager with minimum interval between frames."""
        self.output_dir = output_dir
        self.min_interval = min_interval  # minimum seconds between frames
        self.last_save_time = 0
        self.stage_dirs = {
            'raw': output_dir / 'stage1_raw',
            'filtered': output_dir / 'stage2_filtered'
        }
        self._init_directories()
        
    def _init_directories(self):
        """Initialize directory structure."""
        for dir_path in self.stage_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def _get_frame_path(self, timestamp: int, stage: str = 'raw') -> Path:
        """Generate frame path based on timestamp."""
        dt = datetime.fromtimestamp(timestamp)
        filename = f"{dt.strftime('%Y%m%d_%H%M%S')}.jpg"
        return self.stage_dirs[stage] / filename
        
    def can_save_frame(self, current_time: int) -> bool:
        """Check if enough time has passed since last save."""
        return (current_time - self.last_save_time) >= self.min_interval
        
    def save_frame(self, 
                  frame: np.ndarray,
                  metadata: Dict,
                  current_time: Optional[int] = None,
                  quality: int = 95,
                  save_metadata: bool = True) -> Optional[Path]:
        """Save frame with optional metadata if time threshold is met.
        
        Args:
            frame: The frame to save
            metadata: Metadata dictionary
            current_time: Optional timestamp override
            quality: JPEG quality (0-100)
            save_metadata: Whether to save metadata file (default: True)
        """
        current_time = current_time or int(time.time())
        
        if not self.can_save_frame(current_time):
            return None
            
        # Update last save time
        self.last_save_time = current_time
        
        # Generate paths
        frame_path = self._get_frame_path(current_time)
        
        # Save frame
        cv2.imwrite(
            str(frame_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        
        # Save metadata if enabled
        if save_metadata:
            metadata_path = frame_path.with_suffix('.json')
            # Add timestamp to metadata
            metadata['timestamp'] = current_time
            metadata['datetime'] = datetime.fromtimestamp(current_time).isoformat()
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        return frame_path

def is_similar_frame(frame1: np.ndarray, 
                    frame2: np.ndarray, 
                    threshold: float = 0.95) -> bool:
    """Check if two frames are similar using structural similarity."""
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM between two images
    score = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0][0]
    return score > threshold

def get_timestamp_str() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def resize_frame(frame: np.ndarray, 
                target_width: int, 
                target_height: int) -> np.ndarray:
    """Resize frame while maintaining aspect ratio."""
    height, width = frame.shape[:2]
    
    # Calculate aspect ratio
    aspect = width / height
    target_aspect = target_width / target_height
    
    if aspect > target_aspect:
        # Width is the limiting factor
        new_width = target_width
        new_height = int(target_width / aspect)
    else:
        # Height is the limiting factor
        new_height = target_height
        new_width = int(target_height * aspect)
    
    return cv2.resize(frame, (new_width, new_height)) 