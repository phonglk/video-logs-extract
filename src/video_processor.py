import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
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

from .person_detector import PersonDetector
from .frame_buffer import FrameBuffer
from .utils import (
    get_video_info,
    FrameManager,
    ProgressManager,
    BatchProgressManager,
    resize_frame
)

class VideoProcessor:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the video processor with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configure logging
        logging.basicConfig(
            filename='video_extract.log',
            level=config['processing'].get('log_level', 'INFO'),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Force single thread for OpenCV
        cv2.setNumThreads(1)
        
        # Initialize detector
        self.detector = PersonDetector(config)
        
        # Get processing parameters from config
        self.batch_size = config['processing'].get('batch_size', 32)  # Reduced for better memory usage
        self.frame_skip = config['processing'].get('frame_skip', 5)   # Increased skip rate
        
        # Optimize queue sizes
        config['processing']['frame_queue_size'] = 256  # Reduced for better memory management
        config['processing']['result_queue_size'] = 64  # Reduced for better memory management
        config['processing']['prefetch_batches'] = 2    # Reduced to prevent memory overflow
        
        # Initialize frame manager
        output_dir = Path(config['output']['directory'])
        min_interval = config['output'].get('min_interval_seconds', 5)
        self.frame_manager = FrameManager(output_dir, min_interval)
        
        # Initialize frame buffer with optimized config
        self.frame_buffer = FrameBuffer(config)
        
        # Initialize thread pool for async frame saving
        self.save_pool = ThreadPoolExecutor(max_workers=4)
        self.pending_saves = []
        
        # Performance metrics
        self.perf_metrics = {
            'read_time': [],
            'resize_time': [],
            'detect_time': [],
            'save_time': []
        }

    def __del__(self):
        """Print performance metrics and cleanup."""
        # Wait for pending saves to complete
        if hasattr(self, 'save_pool'):
            self.save_pool.shutdown(wait=True)
            
        if hasattr(self, 'perf_metrics'):
            metrics = self.perf_metrics
            total_batches = len(metrics['read_time'])
            if total_batches > 0:
                print("\nPerformance Metrics (average per batch):")
                print(f"Read time:    {sum(metrics['read_time']) / total_batches:.3f}s")
                print(f"Resize time:  {sum(metrics['resize_time']) / total_batches:.3f}s")
                print(f"Detect time:  {sum(metrics['detect_time']) / total_batches:.3f}s")
                print(f"Save time:    {sum(metrics['save_time']) / total_batches:.3f}s")
                
                # Calculate percentages
                total_time = sum([
                    sum(metrics['read_time']),
                    sum(metrics['resize_time']),
                    sum(metrics['detect_time']),
                    sum(metrics['save_time'])
                ])
                if total_time > 0:
                    print("\nTime distribution:")
                    print(f"Read:    {sum(metrics['read_time']) / total_time * 100:.1f}%")
                    print(f"Resize:  {sum(metrics['resize_time']) / total_time * 100:.1f}%")
                    print(f"Detect:  {sum(metrics['detect_time']) / total_time * 100:.1f}%")
                    print(f"Save:    {sum(metrics['save_time']) / total_time * 100:.1f}%")

    def prepare_frame_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Prepare a batch of frames for processing."""
        if not frames:
            return []
            
        target_size = (
            self.config['detection'].get('resize_width', 640),
            self.config['detection'].get('resize_height', 640)
        )
        
        start_time = time.time()
        
        # Stack frames for batch processing
        # Ensure all frames have the same dimensions
        h, w = frames[0].shape[:2]
        stacked_frames = np.stack(frames)
        
        # Batch resize using OpenCV's optimized functions
        if h > target_size[1] or w > target_size[0]:
            # Downscaling - use AREA interpolation
            prepared_frames = [
                cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                for frame in frames
            ]
        else:
            # Upscaling - use CUBIC interpolation
            prepared_frames = [
                cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC)
                for frame in frames
            ]
            
        self.perf_metrics['resize_time'].append(time.time() - start_time)
        return prepared_frames

    def process_video(self, video_path: str, progress: ProgressManager) -> None:
        """Process a single video file."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return
            
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Extract timestamp from filename (assuming format like "05M44S_1728867944.mp4")
            video_name = Path(video_path).stem
            try:
                video_timestamp = int(video_name.split('_')[1])
                base_time = datetime.fromtimestamp(video_timestamp)
            except (IndexError, ValueError):
                self.logger.warning(f"Could not extract timestamp from filename {video_path}, using current time")
                base_time = datetime.now()
            
            # Calculate actual frames to process accounting for frame skip
            frames_to_process = total_frames // (self.frame_skip + 1) + (total_frames % (self.frame_skip + 1) > 0)
            
            # Log video information
            self.logger.info(f"Processing video: {video_path}")
            self.logger.info(f"Total frames: {total_frames}, Processing frames: {frames_to_process}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
            
            progress.set_total(frames_to_process)
            progress.set_video_info(fps=fps, duration=duration)
            
            # Start frame reading and prefetching
            self.frame_buffer.start_read_thread(cap, self.frame_skip)
            self.frame_buffer.start_prefetch_thread()
            
            batch_start_time = time.time()
            read_start_time = time.time()
            processed_frames = 0
            
            while True:
                frames, frame_numbers = self.frame_buffer.get_batch()
                if not frames:
                    break
                    
                processed_frames += len(frames)
                self.perf_metrics['read_time'].append(time.time() - read_start_time)
                batch_process_time = time.time() - batch_start_time
                
                # Calculate frame timestamps based on frame numbers and FPS
                timestamps = []
                for frame_num in frame_numbers:
                    frame_time = frame_num / fps if fps > 0 else 0
                    ts = base_time + timedelta(seconds=frame_time)
                    timestamps.append(ts)
                
                self._process_batch(frames, timestamps, progress, batch_process_time)
                batch_start_time = time.time()
                read_start_time = time.time()
                
        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {e}")
            
        finally:
            # Cleanup
            try:
                cap.release()
            except:
                pass
                
            try:
                self.frame_buffer.stop()
            except:
                pass
                
            try:
                progress.complete()
            except:
                pass

    def _process_batch(
        self, 
        frames: List[np.ndarray],
        timestamps: List[datetime],
        progress: ProgressManager,
        batch_time: float
    ) -> None:
        """Process a batch of frames with their timestamps."""
        batch_size = len(frames)
        
        # Update batch processing stats
        fps = batch_size / batch_time if batch_time > 0 else 0
        progress.update_batch_stats(fps=fps)
        
        # Prepare frames
        prepared_frames = self.prepare_frame_batch(frames)
        
        # Detect persons in batch
        detect_start_time = time.time()
        detection_results = self.detector.process_batch(prepared_frames)
        self.perf_metrics['detect_time'].append(time.time() - detect_start_time)
        
        # Process results and save frames
        save_start_time = time.time()
        saved_count = 0
        
        # Clean up completed saves
        self.pending_saves = [f for f in self.pending_saves if not f.done()]
        
        for frame, timestamp, result in zip(frames, timestamps, detection_results):
            if result is not None:
                num_persons, detections = result
                
                # Save frame using frame manager
                metadata = {
                    'detections': detections,
                    'num_persons': num_persons,
                    'frame_time': timestamp.isoformat()
                }
                
                # Submit frame save to thread pool
                future = self.save_pool.submit(
                    self.frame_manager.save_frame,
                    frame.copy(),  # Create a copy for async saving
                    metadata,
                    int(timestamp.timestamp()),
                    self.config['output'].get('quality', 95)
                )
                self.pending_saves.append(future)
                saved_count += 1
                
        self.perf_metrics['save_time'].append(time.time() - save_start_time)
            
        # Update progress with batch results
        progress.update(
            processed=batch_size,
            saved=saved_count,
            batch_size=batch_size
        )

    def process_directory(self, input_dir: str) -> None:
        """Process all videos in a directory."""
        input_path = Path(input_dir)
        supported_formats = self.config['input'].get(
            'supported_formats',
            ['.mp4', '.avi', '.mov']
        )
        
        # Find all video files
        video_files = []
        for fmt in supported_formats:
            video_files.extend(input_path.rglob(f'*{fmt}'))
            
        if not video_files:
            self.logger.warning(f"No supported video files found in {input_dir}")
            return
            
        total_videos = len(video_files)
        print(f"\nFound {total_videos} videos to process")
        
        # Initialize batch progress manager
        batch_manager = BatchProgressManager(total_videos)
        
        # Process each video
        for i, video_file in enumerate(video_files, 1):
            print(f"\nProcessing video {i}/{total_videos}: {video_file.name}")
            progress = ProgressManager(batch_manager=batch_manager)
            self.process_video(str(video_file), progress)
            batch_manager.video_complete()
            
        # Cleanup
        print("\nBatch processing complete!") 