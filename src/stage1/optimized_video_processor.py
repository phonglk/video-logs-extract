import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Set
import cv2
import numpy as np
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
from datetime import datetime, timedelta
import json
import multiprocessing as mp
import psutil
import threading
import queue
import yaml
import heapq
from collections import deque
import logging.handlers

# Import the optimized components
from ..optimized_processor import OptimizedFrameBuffer, parallel_resize, OptimizedVideoProcessor as BaseOptimizedProcessor

# Import original stage1 components
from .person_detector import PersonDetector
from .utils import (
    get_video_info,
    FrameManager,
    ProgressManager,
    BatchProgressManager
)

class OptimizedVideoProcessor(BaseOptimizedProcessor):
    """
    Stage1 specialized video processor that extends the base OptimizedVideoProcessor
    with stage1-specific functionality like person detection and frame extraction.
    """
    def __init__(self, config: Dict[str, Any]):
        """Initialize the video processor with configuration."""
        # Initialize base processor
        super().__init__(config)
        
        # Configure file logging if enabled in config
        log_file = config['processing'].get('log_file')
        if log_file:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                
            # Set up file handler with rotation
            try:
                # Configure file handler with rotation (10 MB max size, keep 5 backup files)
                max_bytes = config['processing'].get('log_max_bytes', 10 * 1024 * 1024)  # 10 MB default
                backup_count = config['processing'].get('log_backup_count', 5)  # 5 backups default
                
                # Use RotatingFileHandler for log rotation
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count
                )
                
                # Set the formatter
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                
                # Set the log level for the file handler (can be different from console)
                file_log_level = config['processing'].get('file_log_level', self.logger.level)
                if isinstance(file_log_level, str):
                    file_log_level = getattr(logging, file_log_level)
                file_handler.setLevel(file_log_level)
                
                # Add the handler to the logger
                self.logger.addHandler(file_handler)
                
                # Also add the handler to the root logger to capture logs from all modules
                root_logger = logging.getLogger()
                root_logger.addHandler(file_handler)
                
                self.logger.info(f"Log file configured: {log_file}")
            except Exception as e:
                self.logger.error(f"Failed to set up log file handler: {e}")
        
        # Configure stage1-specific settings
        
        # Set up output directories for stage1
        output_dir = Path(config['output'].get('directory', 'output'))
        
        # Get stage1-specific output directories from config
        stage1_config = config.get('stage1', {})
        raw_dir_name = stage1_config.get('raw_dir', 'raw')
        
        self.raw_dir = output_dir / raw_dir_name
        os.makedirs(self.raw_dir, exist_ok=True)
        
        # Store the directory paths in config for other components or stages to access
        if 'stage1' not in config:
            config['stage1'] = {}
        config['stage1']['raw_dir'] = str(self.raw_dir)
        
        # Initialize frame manager for stage1
        min_interval = config['output'].get('min_interval_seconds', 5)
        self.frame_manager = FrameManager(output_dir, min_interval)
        
        # Initialize detector for stage1 (lazy loading - will be created when needed)
        self.detector = None
        
        # Set up resume support for stage1
        self.resume_file = Path(config.get('stage1', {}).get('resume_file', 'data/stage1_output.json'))
        
        # Track processing times for percentile calculation
        self.processing_times = []
        
        # Skip resume if disabled in config
        if config.get('resume', True):
            self.processed_videos, self.processing_times = self._load_processed_videos()
            self.logger.info(f"Resume enabled, loaded {len(self.processed_videos)} processed videos")
        else:
            self.processed_videos = set()
            self.processing_times = []
            self.logger.info("Resume disabled, starting fresh")
            
    def _load_processed_videos(self) -> Tuple[Set[str], List[float]]:
        """Load list of already processed videos and their processing times for resume support."""
        try:
            if self.resume_file.exists():
                with open(self.resume_file) as f:
                    data = json.load(f)
                    videos = set(data.get('processed_videos', []))
                    times = data.get('processing_times', [])
                    self.logger.info(f"Loaded {len(videos)} processed videos from resume file")
                    return videos, times
            return set(), []
        except Exception as e:
            self.logger.error(f"Error loading resume file: {e}")
            return set(), []

    def _save_progress(self, video_path: str, processing_time: float) -> None:
        """Save video path and processing time to the processed list for resume support."""
        try:
            self.processed_videos.add(video_path)
            
            # Add processing time to the list
            self.processing_times.append(processing_time)
            
            # Keep only the last 100 processing times
            if len(self.processing_times) > 100:
                self.processing_times = self.processing_times[-100:]
            
            # Calculate 90th percentile of processing times
            if self.processing_times:
                # Sort the processing times to compute percentile
                sorted_times = sorted(self.processing_times)
                p90_index = int(0.9 * len(sorted_times))
                p90_time = sorted_times[p90_index]
            else:
                p90_time = 0
            
            # Ensure directory exists
            self.resume_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(self.resume_file, 'w') as f:
                json.dump({
                    'processed_videos': list(self.processed_videos),
                    'last_processed': video_path,
                    'timestamp': datetime.now().isoformat(),
                    'count': len(self.processed_videos),
                    'processing_times': self.processing_times,
                    'p90_processing_time': p90_time
                }, f, indent=2)
                
            self.logger.info(f"Updated progress file with {len(self.processed_videos)} processed videos")
            self.logger.info(f"90th percentile processing time of last {len(self.processing_times)} videos: {p90_time:.2f}s")
        except Exception as e:
            self.logger.error(f"Error saving progress: {e}")
    
    # Override _save_frame_and_metadata to use stage1-specific logic
    def _save_frame_and_metadata(self, frame: np.ndarray, detections: List[Dict], frame_index: int = None) -> None:
        """Save a frame and its metadata using stage1-specific format."""
        save_start = time.time()
        
        try:
            # Get video timestamp if we have extracted one from filename, otherwise use current time
            video_timestamp = getattr(self, 'video_timestamp', None)
            video_fps = getattr(self, 'video_fps', None)
            
            # Calculate frame timestamp based on video start time and frame index if possible
            if video_timestamp and video_fps and frame_index is not None:
                # Calculate seconds from start of video to this frame
                frame_time_seconds = frame_index / video_fps
                # Add this offset to the video start timestamp
                timestamp = video_timestamp + int(frame_time_seconds)
            else:
                # Fallback to current time
                timestamp = int(datetime.now().timestamp())
            
            # Convert the frame back to BGR for saving with OpenCV
            # Note: If frames were converted to RGB in _prepare_batch, we need to convert back
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Get configuration settings
            output_format = self.config['output'].get('format', '{timestamp}.jpg')
            jpeg_quality = self.config['output'].get('quality', 90)
            include_metadata = self.config['output'].get('include_metadata', False)
            metadata_format = self.config['output'].get('metadata_format', '{timestamp}.json')
            
            # Format the filename with timestamp
            dt = datetime.fromtimestamp(timestamp)
            timestamp_str = dt.strftime('%Y%m%d_%H%M%S')
            
            # Replace {timestamp} with actual timestamp
            frame_filename = output_format.replace('{timestamp}', timestamp_str)
            frame_path = self.raw_dir / frame_filename
            
            # Save frame with configured quality
            cv2.imwrite(
                str(frame_path),
                frame_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            
            # Save metadata if enabled
            if include_metadata:
                metadata_filename = metadata_format.replace('{timestamp}', timestamp_str)
                metadata_path = self.raw_dir / metadata_filename
                
                # Include frame information in metadata
                metadata = {
                    "timestamp": timestamp,
                    "datetime": dt.isoformat(),
                    "frame_index": frame_index,
                    "detections": detections
                }
                
                # Write metadata to file
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
            save_time = time.time() - save_start
            self.perf_metrics["save_time"] += save_time
            
            # Log successful save
            self.logger.debug(f"Saved frame {frame_index} to {frame_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving frame: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    # Override _process_batch to use stage1-specific detector
    def _process_batch(self, batch_indices: List[int], batch_frames: List[np.ndarray]):
        """Process a batch of frames using stage1 person detector."""
        # Debug logging
        self.logger.info(f"Processing batch with {len(batch_frames)} frames")
        
        # Initialize detector on first use
        if self.detector is None:
            self.logger.info("Initializing PersonDetector")
            try:
                self.detector = PersonDetector(self.config)
                self.logger.info("PersonDetector initialized successfully")
            except Exception as e:
                self.logger.error(f"Error initializing PersonDetector: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # Return empty results if we can't initialize the detector
                return [None] * len(batch_frames)
            
        # Run detection
        detect_start = time.time()
        self.logger.info(f"Running detection on batch of {len(batch_frames)} frames")
        try:
            # Add debug info about frames
            if batch_frames:
                sample_frame = batch_frames[0]
                self.logger.info(f"Sample frame shape: {sample_frame.shape}, dtype: {sample_frame.dtype}")
                
            # Validate frames before sending to detector
            valid_frames = []
            valid_indices = []
            invalid_count = 0
            
            for i, frame in enumerate(batch_frames):
                if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                    invalid_count += 1
                    continue
                valid_frames.append(frame)
                valid_indices.append(i)
                
            if invalid_count > 0:
                self.logger.warning(f"Skipped {invalid_count} invalid frames before detection")
                
            if not valid_frames:
                self.logger.error("No valid frames for detection")
                return [None] * len(batch_frames)
                
            # Process valid frames using stage1-specific detector
            valid_results = self.detector.process_batch(valid_frames)
            
            # Map valid results back to original batch positions
            batch_results = [None] * len(batch_frames)
            for i, orig_idx in enumerate(valid_indices):
                if i < len(valid_results):
                    batch_results[orig_idx] = valid_results[i]
            
            detect_time = time.time() - detect_start
            detection_count = sum(1 for r in batch_results if r is not None)
            
            self.logger.info(f"Detection completed in {detect_time:.2f}s, found {detection_count} frames with detections")
            
            self.perf_metrics["detection_time"] += detect_time
        except Exception as e:
            self.logger.error(f"Error during detection: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return [None] * len(batch_frames)
        
        # Optimization: Use a local counter for tracking detections in this batch
        # to avoid constant recomputing the sum
        frames_with_detections = 0
        
        # Group the save operations to reduce file system pressure
        save_operations = []
        
        # First collect all save operations
        for i, (frame_index, frame) in enumerate(zip(batch_indices, batch_frames)):
            try:
                result = batch_results[i]
                
                # Skip frames with no detections
                if result is None:
                    continue
                    
                # Get the detections for this frame
                num_persons, detections = result
                
                frames_with_detections += 1
                self.logger.debug(f"Frame {frame_index}: Detected {num_persons} persons")
                
                # Make a copy of the frame for async saving
                frame_copy = frame.copy()
                
                # Add to save operations list instead of submitting immediately
                save_operations.append((frame_copy, detections, frame_index))
                
            except Exception as e:
                self.logger.error(f"Error processing detection result for frame {i}: {e}")
                
        # Submit all save operations at once and manage thread pool more efficiently
        if save_operations:
            self.logger.info(f"Submitting {len(save_operations)} frames for saving")
            
            # Process save operations in smaller groups to avoid memory spikes
            max_concurrent_saves = min(8, len(save_operations))  # Limit concurrent saves
            
            # Only create new futures if we have space
            current_pending = len(self.pending_saves)
            space_available = max(0, 16 - current_pending)  # Keep pending saves under 16
            
            # If we have limited space, prioritize the first few frames
            if space_available < len(save_operations):
                save_operations = save_operations[:space_available]
                self.logger.info(f"Limiting save operations to {space_available} due to backlog")
                
            # Submit save operations
            new_futures = []
            for frame, detections, frame_index in save_operations:
                future = self.save_pool.submit(
                    self._save_frame_and_metadata,
                    frame,
                    detections,
                    frame_index
                )
                new_futures.append(future)
                
            # Add to pending saves list
            self.pending_saves.extend(new_futures)
        
        # Optimize pending save cleanup - only clean if we have many pending
        if len(self.pending_saves) > 32:
            # Clean up completed saves more efficiently
            self.pending_saves = [f for f in self.pending_saves if not f.done()]
            self.logger.info(f"Cleaned up completed saves, {len(self.pending_saves)} still pending")
        
        self.logger.info(f"Saving {frames_with_detections} frames with detections")
        return batch_results
    
    # Override process_video to include stage1-specific resume functionality
    def process_video(self, video_path: str, progress: Optional[ProgressManager] = None, 
                   skip_logging: bool = False) -> bool:
        """Process a single video file with stage1-specific handling."""
        self.logger.info(f"Processing video: {video_path}")
        
        # Check if this video was already processed (for resume support)
        video_path_str = str(video_path)
        if video_path_str in self.processed_videos:
            # Skip verbose logging if requested (for batch processing)
            if not skip_logging:
                # Log as debug instead of printing to stdout
                self.logger.debug(f"Skipping already processed video: {video_path}")
            return False
        
        try:
            # Reset frame buffer for new video
            self.frame_buffer.reset()
            
            # Extract timestamp from filename (optional)
            video_name = os.path.basename(video_path)
            self.video_timestamp = self._extract_timestamp(video_name)
            if self.video_timestamp:
                self.logger.info(f"Extracted timestamp from filename: {self.video_timestamp} "
                               f"({datetime.fromtimestamp(self.video_timestamp).strftime('%Y-%m-%d %H:%M:%S')})")
            else:
                self.logger.warning(f"Could not extract timestamp from filename: {video_name}, "
                                 f"will use current time for frame timestamps")
            
            # Start processing
            processing_start = time.time()
            
            # Start frame buffer threads
            if not self.frame_buffer.start_read_thread(video_path):
                self.logger.error(f"Failed to start frame reading for {video_path}")
                return False
                
            self.frame_buffer.start_prefetch_thread()
            
            # Get video properties
            try:
                import cv2
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    self.video_fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / self.video_fps if self.video_fps > 0 else 0
                    
                    # Pass the filename to the progress manager if it exists
                    if progress is not None:
                        progress.set_video_info(self.video_fps, duration, video_path)
                        
                    cap.release()
                    self.logger.info(f"Video FPS: {self.video_fps}, Duration: {duration:.1f} seconds")
                else:
                    self.video_fps = None
                    self.logger.warning("Could not open video to get FPS")
            except Exception as e:
                self.video_fps = None
                self.logger.warning(f"Could not get video info: {e}")
            
            # Process batches
            frames_processed = 0
            frames_saved = 0
            
            # Add debugging to track frame processing
            batch_count = 0
            self.logger.info("Starting batch processing loop")
            
            # Create progress manager if not provided
            if progress is None:
                progress = ProgressManager()
                if hasattr(self, 'video_fps') and self.video_fps and video_path:
                    try:
                        # Set video info in the newly created progress manager
                        duration = frame_count / self.video_fps if self.video_fps > 0 else 0
                        progress.set_video_info(self.video_fps, duration, video_path)
                    except Exception as e:
                        self.logger.warning(f"Could not set video info on new progress manager: {e}")
                progress.set_total(self.frame_buffer.frames_to_process)
            else:
                # Update total frames in existing progress manager
                progress.set_total(self.frame_buffer.frames_to_process)
            
            # Start progress tracking
            progress_start_time = time.time()
            
            while True:
                batch_indices, batch_frames = self.frame_buffer.get_batch()
                
                if batch_indices is None or not batch_frames:
                    self.logger.info(f"No more batches to process after {batch_count} batches")
                    break
                    
                batch_count += 1
                self.logger.info(f"Processing batch {batch_count}: {len(batch_frames)} frames")
                
                # Process the batch
                detections = self._process_batch(batch_indices, batch_frames)
                
                # Update counters
                frames_processed += len(batch_frames)
                saved_in_batch = sum(1 for d in detections if d is not None)
                frames_saved += saved_in_batch
                
                self.logger.info(f"Batch {batch_count} results: {saved_in_batch}/{len(batch_frames)} frames saved")
                
                # Update progress
                progress.update(processed=len(batch_frames), saved=saved_in_batch)
                
                # Log progress
                progress_percent = min(100, int(frames_processed / self.frame_buffer.frames_to_process * 100))
                elapsed = time.time() - processing_start
                eta = (elapsed / frames_processed) * (self.frame_buffer.frames_to_process - frames_processed) if frames_processed > 0 else 0
                
                self.logger.info(f"Progress: {progress_percent}% ({frames_processed}/{self.frame_buffer.frames_to_process} frames), "
                               f"Saved: {frames_saved} frames, ETA: {eta:.1f}s")
                
            # Finalize progress
            progress.complete()
            
            # Update performance metrics
            total_time = time.time() - processing_start
            buffer_stats = self.frame_buffer.get_statistics()
            
            self.perf_metrics["total_time"] += total_time
            self.perf_metrics["read_time"] += buffer_stats["read_time"]
            self.perf_metrics["resize_time"] += buffer_stats["resize_time"]
            self.perf_metrics["frames_processed"] += frames_processed
            self.perf_metrics["frames_saved"] += frames_saved
            
            self.logger.info(f"Video processing completed: {frames_processed} frames processed, "
                           f"{frames_saved} frames saved in {total_time:.2f}s")
            
            # Mark as processed for resume support
            self._save_progress(video_path_str, total_time)
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
            
        finally:
            # Clean up resources for this video
            self.frame_buffer.stop()
    
    # Add stage1-specific directory processing method 
    def process_directory(self, input_dir: str) -> None:
        """Process all videos in a directory with stage1-specific handling."""
        input_path = Path(input_dir)
        self.logger.info(f"Processing directory: {input_path}")
        
        # Get list of supported formats
        supported_formats = self.config.get('supported_formats', ['.mp4', '.avi', '.mkv', '.mov'])
        
        # Find all video files
        video_files = []
        for fmt in supported_formats:
            video_files.extend(input_path.glob(f"**/*{fmt}"))
            
        if not video_files:
            self.logger.warning(f"No video files found in {input_path}")
            return
            
        # Count total already processed videos
        skipped_count = 0
        processed_count = 0
        error_count = 0
        
        self.logger.info(f"Found {len(video_files)} video files to process")
        print(f"\nProcessing {len(video_files)} videos...")
        
        # Process each video
        for idx, video_file in enumerate(video_files):
            # Create progress manager for this video
            progress = ProgressManager()
            progress.set_total(100)
            
            try:
                # Process the video with skip_logging=True to avoid logging each skipped file
                was_processed = self.process_video(str(video_file), progress, skip_logging=True)
                
                if not was_processed:
                    skipped_count += 1
                else:
                    processed_count += 1
                    
                # Print overall progress periodically (every 10 videos or at the end)
                if (idx + 1) % 10 == 0 or idx == len(video_files) - 1:
                    total_handled = idx + 1
                    print(f"\rProgress: {total_handled}/{len(video_files)} videos | "
                          f"Processed: {processed_count} | Skipped: {skipped_count} | "
                          f"Errors: {error_count}", end="")
                    
            except Exception as e:
                self.logger.error(f"Error processing {video_file}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                error_count += 1
                continue
                
        # Print final summary
        print(f"\n\nProcessing complete:")
        print(f"  - Videos found: {len(video_files)}")
        print(f"  - Processed successfully: {processed_count}")
        if skipped_count > 0:
            print(f"  - Skipped (already processed): {skipped_count}")
        if error_count > 0:
            print(f"  - Failed with errors: {error_count}")
        self.logger.warning("=" * 50)
        self.logger.warning(f"Completed processing directory with {processed_count} processed, {skipped_count} skipped, {error_count} errors")
        self.logger.warning("=" * 50)

def create_from_config(config_path: str) -> OptimizedVideoProcessor:
    """Create an optimized video processor from a config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return OptimizedVideoProcessor(config) 