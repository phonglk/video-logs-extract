#!/usr/bin/env python
import time
import logging
import os
import cv2
import numpy as np
import threading
import queue
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import psutil
import multiprocessing as mp
import yaml
import argparse

# Add ProcessPoolExecutor import at the top level
try:
    from concurrent.futures import ProcessPoolExecutor
except ImportError:
    ProcessPoolExecutor = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("optimized_processor")

# Add this at the module level, before the classes
def parallel_resize(args):
    """
    Helper function for parallel frame resizing.
    Args are (frame, size, interpolation)
    """
    frame, size, interpolation = args
    try:
        # Ensure the frame is not None and has the correct shape
        if frame is None:
            return None
            
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            # Log error or handle invalid frame
            return None
            
        return cv2.resize(frame, size, interpolation=interpolation)
    except Exception as e:
        # Return the original frame if resizing fails
        return frame

class OptimizedFrameBuffer:
    """
    Optimized frame buffer for reading, buffering, and preparing frames 
    with improved memory management and performance.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_size = config['processing'].get('batch_size', 32)
        # Use more aggressive frame skipping for high-res videos
        self.frame_skip = config['processing'].get('frame_skip', 10)
        self.width = config['detection'].get('resize_width', 640)
        self.height = config['detection'].get('resize_height', 640)
        
        # Improved queue sizing based on memory constraints
        # For high-res videos, keep queue sizes smaller to avoid memory pressure
        # These queue settings are optimized for 2K+ resolution videos
        self.frame_queue_size = 64  # Reduced from 256 to avoid memory pressure
        self.result_queue_size = 32
        self.prefetch_batches = 4   # Smaller prefetch to reduce memory usage
        
        # Calculate approx memory usage per frame: width * height * 3 bytes (RGB) + overhead
        self.approx_frame_memory = (self.width * self.height * 3) * 1.2  # 20% overhead
        
        # Set available memory target (90% of 4GB or system memory if lower)
        system_memory_gb = psutil.virtual_memory().total / (1024**3) if 'psutil' in globals() else 16
        target_memory_gb = min(system_memory_gb * 0.4, 4.0)  # Use 40% of system memory up to 4GB
        target_memory = target_memory_gb * 1024**3
        
        # Adjust queue sizes based on memory constraints
        max_frames_in_memory = int(target_memory / self.approx_frame_memory)
        self.frame_queue_size = min(self.frame_queue_size, max(16, int(max_frames_in_memory * 0.4)))
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Frame buffer initialized with frame_skip={self.frame_skip}, "
                        f"queue_size={self.frame_queue_size}, target_size={self.width}x{self.height}")
        
        # Initialize stopped flag and events
        self.stopped = False
        self.stop_event = threading.Event()
        
        # Initialize thread references
        self.read_thread = None
        self.prefetch_thread = None
        
        # Initialize queues with deque-based implementation for faster operations
        self.frame_queue = queue.Queue(maxsize=self.frame_queue_size)
        self.prefetch_queue = queue.Queue(maxsize=self.prefetch_batches)
        self.result_queue = queue.Queue(maxsize=self.result_queue_size)
        
        # Check for hardware acceleration availability
        self.has_hw_accel = self._check_hw_acceleration()
        
        # Statistics
        self.stats = {
            "read_time": 0,
            "resize_time": 0,
            "read_frames": 0,
            "skipped_frames": 0
        }
    
    def _check_hw_acceleration(self) -> bool:
        """Check if hardware acceleration is available for decoding."""
        # Check for hardware acceleration options
        codec_options = {
            # MacOS options
            'videotoolbox': cv2.videoreader_qt.getBackends() if hasattr(cv2, 'videoreader_qt') else [],
            # NVIDIA options
            'cuda': [cv2.cuda.getCudaEnabledDeviceCount() > 0] if hasattr(cv2, 'cuda') else [],
            # Intel options
            'qsv': hasattr(cv2, 'VideoCapture_INTEL'),
            # General options 
            'any': cv2.getBuildInformation().find('ffmpeg') != -1
        }
        
        has_accel = any(codec_options.values())
        self.logger.info(f"Hardware acceleration available: {has_accel}")
        return has_accel
        
    def reset(self):
        """Reset buffer for new video processing."""
        # Stop existing threads if running
        if not self.stopped:
            self.stop()
            
        # Reset queues
        self.frame_queue = queue.Queue(maxsize=self.frame_queue_size)
        self.prefetch_queue = queue.Queue(maxsize=self.prefetch_batches)
        self.result_queue = queue.Queue(maxsize=self.result_queue_size)
        
        # Reset state
        self.stopped = False
        self.stop_event.clear()
        
        # Reset statistics
        self.stats = {
            "read_time": 0,
            "resize_time": 0,
            "read_frames": 0,
            "skipped_frames": 0
        }
    
    def start_read_thread(self, video_path: str):
        """Start thread to read frames from the video."""
        if self.read_thread is not None and self.read_thread.is_alive():
            return
            
        # Try to open with hardware acceleration if available
        if self.has_hw_accel and hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
            self.cap = cv2.VideoCapture(video_path)
            # Try available hardware accelerations
            accel_options = [cv2.VIDEO_ACCELERATION_ANY, cv2.VIDEO_ACCELERATION_D3D11, 
                         cv2.VIDEO_ACCELERATION_VAAPI, cv2.VIDEO_ACCELERATION_MFX]
            
            for accel in accel_options:
                self.cap.release()  # Release before trying a new option
                self.cap = cv2.VideoCapture(video_path)
                self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, accel)
                if self.cap.isOpened():
                    self.logger.info(f"Opened video with hardware acceleration: {accel}")
                    break
        else:
            # Fall back to standard opening
            self.cap = cv2.VideoCapture(video_path)
            
        if not self.cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return False
        
        # Try to use more efficient video decoding settings
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Small buffer to reduce memory usage
        
        # Get total frames, FPS, and calculate processing frames
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames_to_process = (self.total_frames + self.frame_skip) // (self.frame_skip + 1)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Adaptive frame skipping based on resolution
        if self.config['detection'].get('adaptive_skip', True):
            # For very high res videos (>2K), increase skip rate
            if self.frame_width * self.frame_height > 2000000:
                original_skip = self.frame_skip
                self.frame_skip = max(self.frame_skip, 15)  # At least skip 15 frames
                self.logger.info(f"High resolution video detected ({self.frame_width}x{self.frame_height}), "
                                f"increasing frame skip from {original_skip} to {self.frame_skip}")
                self.frames_to_process = (self.total_frames + self.frame_skip) // (self.frame_skip + 1)
        
        self.logger.info(f"Starting read thread for video with {self.total_frames} frames "
                        f"({self.frame_width}x{self.frame_height} @ {self.fps:.1f} FPS) "
                        f"processing ~{self.frames_to_process} frames with frame_skip={self.frame_skip}")
        
        self.read_thread = threading.Thread(target=self._read_frames, daemon=True)
        self.read_thread.start()
        return True
        
    def _read_frames(self):
        """Thread function to read frames from video."""
        frame_index = 0
        processed_frames = 0
        skipped_frames = 0
        self.logger.info("Starting read thread")
        
        try:
            while not self.stop_event.is_set():
                read_start = time.time()
                ret, frame = self.cap.read()
                read_time = time.time() - read_start
                
                if not ret:
                    self.logger.info(f"Read thread: End of video reached after {frame_index} frames")
                    break
                    
                self.stats["read_time"] += read_time
                self.stats["read_frames"] += 1
                
                # Process this frame if it's on our frame_skip interval
                if frame_index % (self.frame_skip + 1) == 0:
                    try:
                        # Memory optimization: pre-resize high-resolution frames before putting in queue
                        # This significantly reduces memory pressure for 2K+ videos
                        if self.frame_width > 1920 or self.frame_height > 1080:
                            # Scale down by 50% for initial queue storage
                            scale_factor = 0.5
                            small_width = int(self.frame_width * scale_factor)
                            small_height = int(self.frame_height * scale_factor)
                            frame = cv2.resize(frame, (small_width, small_height), 
                                            interpolation=cv2.INTER_AREA)
                            
                        # Add timeout to avoid blocking forever
                        self.logger.debug(f"Read thread: Putting frame {frame_index} in queue")
                        self.frame_queue.put((frame_index, frame), timeout=0.5)
                        processed_frames += 1
                    except queue.Full:
                        self.logger.warning("Frame queue full, dropping frame")
                else:
                    skipped_frames += 1
                        
                frame_index += 1
                
                # Log progress periodically
                if frame_index % 100 == 0:
                    self.logger.info(f"Read thread: Processed {frame_index} frames, "
                                    f"kept {processed_frames}, skipped {skipped_frames}")
                
                # Check every 50 frames if we should stop
                if frame_index % 50 == 0 and self.stop_event.is_set():
                    self.logger.info("Read thread: Stop event set, breaking")
                    break
                    
                # Add a small sleep to reduce CPU usage when queue is nearly full
                if self.frame_queue.qsize() > self.frame_queue_size * 0.8:
                    time.sleep(0.001)  # 1ms sleep when queue is getting full
        except Exception as e:
            self.logger.error(f"Error in read thread: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            # Update stats for skipped frames
            self.stats["skipped_frames"] = skipped_frames
            
            # Signal we're done by putting None in the queue
            try:
                self.logger.info("Read thread: Putting None in frame queue to signal end")
                self.frame_queue.put(None, timeout=1.0)
            except queue.Full:
                self.logger.error("Read thread: Frame queue full when trying to put None")
                
            self.logger.info(f"Read thread finished after processing {frame_index} frames, "
                           f"kept {processed_frames}, skipped {skipped_frames}")
            
    def start_prefetch_thread(self):
        """Start thread to prefetch and prepare batches of frames."""
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            return
            
        self.prefetch_thread = threading.Thread(target=self._prefetch_frames, daemon=True)
        self.prefetch_thread.start()
        
    def _prefetch_frames(self):
        """Thread function to prefetch and prepare batches of frames."""
        self.logger.info("Starting prefetch thread")
        total_batches = 0
        try:
            while not self.stop_event.is_set():
                batch_frames = []
                batch_indices = []
                
                # Collect frames for a batch
                self.logger.info(f"Prefetch thread: Collecting frames for a batch, target size={self.batch_size}")
                frame_collect_start = time.time()
                
                # Memory optimized collection
                collect_timeout = 0.5  # Initial timeout
                while len(batch_frames) < self.batch_size:
                    try:
                        # Add timeout to avoid blocking forever
                        # Use a short timeout initially, longer for subsequent attempts
                        item = self.frame_queue.get(timeout=collect_timeout)
                        self.frame_queue.task_done()
                        
                        # Shorten timeout after first successful get to make batch collection faster
                        collect_timeout = 0.1
                        
                        # Check if we've reached the end of the video
                        if item is None:
                            self.logger.info("Prefetch thread: Received None, end of video reached")
                            break
                            
                        frame_index, frame = item
                        
                        # Skip None or corrupt frames
                        if frame is None or frame.size == 0:
                            self.logger.warning(f"Skipping invalid frame at index {frame_index}")
                            continue
                            
                        batch_frames.append(frame)
                        batch_indices.append(frame_index)
                        
                        if len(batch_frames) % 10 == 0:
                            self.logger.info(f"Prefetch thread: Collected {len(batch_frames)}/{self.batch_size} frames so far")
                    except queue.Empty:
                        # If the queue is empty for a while, we might be done
                        self.logger.info("Prefetch thread: Frame queue empty, checking if read thread is still active")
                        if self.stop_event.is_set() or not self.read_thread.is_alive():
                            self.logger.info("Prefetch thread: Read thread finished or stop event set, breaking")
                            break
                            
                        # Check if we have enough frames to proceed with a smaller batch
                        if len(batch_frames) >= max(4, self.batch_size // 2):
                            self.logger.info(f"Prefetch thread: Proceeding with partial batch of {len(batch_frames)} frames")
                            break
                
                frame_collect_time = time.time() - frame_collect_start
                self.logger.info(f"Prefetch thread: Collected {len(batch_frames)} frames in {frame_collect_time:.2f}s")
                            
                # If we have no frames, we're done
                if not batch_frames:
                    self.logger.info("Prefetch thread: No frames collected, exiting")
                    break
                    
                # Prepare the batch (resize frames)
                prep_start = time.time()
                self.logger.info(f"Prefetch thread: Preparing batch of {len(batch_frames)} frames")
                prepared_batch = self._prepare_batch(batch_frames)
                prep_time = time.time() - prep_start
                self.stats["resize_time"] += prep_time
                
                # Calculate average resize time per frame
                avg_resize_time = prep_time / len(batch_frames) if batch_frames else 0
                self.logger.info(f"Prefetch thread: Batch prepared in {prep_time:.2f}s, "
                               f"avg {avg_resize_time*1000:.1f}ms per frame")
                
                # Put prepared batch in prefetch queue
                try:
                    self.logger.info("Prefetch thread: Putting batch in prefetch queue")
                    self.prefetch_queue.put((batch_indices, prepared_batch), timeout=1.0)
                    self.logger.info("Prefetch thread: Batch successfully put in prefetch queue")
                    total_batches += 1
                except queue.Full:
                    self.logger.warning("Prefetch queue full, dropping batch")
                    
        except Exception as e:
            self.logger.error(f"Error in prefetch thread: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            # Signal we're done
            try:
                self.logger.info("Prefetch thread: Putting None in prefetch queue to signal end")
                self.prefetch_queue.put(None, timeout=1.0)
            except queue.Full:
                self.logger.error("Prefetch thread: Prefetch queue full when trying to put None")
                
            self.logger.info(f"Prefetch thread finished after preparing {total_batches} batches")
            
    def _prepare_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Prepare a batch of frames for processing.
        Optimized batch resizing using parallelization when enabled.
        """
        if not frames:
            return []
        
        # Detect if frames are pre-resized in read thread and need less processing
        if frames and frames[0] is not None:
            h, w = frames[0].shape[:2]
            pre_resized = (w < self.frame_width or h < self.frame_height)
            # Log if frames are already pre-resized
            if pre_resized:
                self.logger.debug(f"Processing pre-resized frames: {w}x{h} -> {self.width}x{self.height}")
        else:
            pre_resized = False
        
        # Get batch size for more efficient parallel processing
        batch_size = len(frames)
        parent = self.config.get('parent')
        
        # Only use multiprocessing for larger batches and when high quality is needed
        use_multiprocessing = (parent and 
                             hasattr(parent, 'use_multiprocessing') and 
                             parent.use_multiprocessing and
                             batch_size >= 8 and
                             not pre_resized)  # Don't use multiprocessing for pre-resized frames
        
        # Added preprocessing step: convert all frames from BGR to RGB
        # since YOLO expects RGB format and OpenCV reads in BGR
        frames_rgb = []
        for frame in frames:
            if frame is None:
                self.logger.warning("Found None frame in batch, skipping")
                continue
            
            if len(frame.shape) < 3 or frame.shape[2] != 3:
                self.logger.warning(f"Invalid frame shape: {frame.shape}, expected 3 channels")
                continue
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_rgb.append(frame_rgb)
        
        if not frames_rgb:
            self.logger.warning("No valid frames after preprocessing")
            return []
            
        frames = frames_rgb  # Use RGB frames for further processing
        
        # For high-resolution, pre-resized frames, use fast but lower quality resize
        # Only when downscaling by a large factor, as we already did initial resize
        if pre_resized:
            self.logger.debug("Using faster INTER_NEAREST resize for pre-resized frames")
            resized_frames = [cv2.resize(frame, (self.width, self.height), 
                                      interpolation=cv2.INTER_NEAREST) 
                            for frame in frames]
            self.logger.info(f"Prepared {len(resized_frames)} frames for detection (fast mode)")
            return resized_frames
                             
        if use_multiprocessing:
            # Use the parent's process pool for parallel resizing with smaller chunks
            # to better utilize multicore performance
            resize_tasks = []
            
            # Group frames into chunks for better throughput
            chunk_size = max(1, min(4, batch_size // parent.resize_workers))
            
            # Prepare tasks in chunks for better processing efficiency
            for frame in frames:
                h, w = frame.shape[:2]
                interpolation = cv2.INTER_AREA if (w > self.width or h > self.height) else cv2.INTER_LINEAR
                resize_tasks.append((frame, (self.width, self.height), interpolation))
            
            # Execute resize tasks in parallel - we need to better manage memory here
            # so we'll use our own implementation instead of map() which loads all results at once
            try:
                futures = []
                for task in resize_tasks:
                    futures.append(parent.resize_pool.submit(parallel_resize, task))
                
                # Collect results as they complete - use a timeout to avoid hanging
                resized_frames = []
                for future in futures:
                    try:
                        result = future.result(timeout=2.0)  # 2 second timeout per frame
                        resized_frames.append(result)
                    except Exception as e:
                        self.logger.error(f"Error in parallel resize: {e}")
                        # Add a placeholder frame in case of error
                        blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                        resized_frames.append(blank)
                        
                # Clear pending tasks
                parent.pending_resizes = []
            except Exception as e:
                self.logger.error(f"Error in parallel frame resizing: {e}")
                # Fall back to sequential processing
                resized_frames = []
                for frame in frames:
                    h, w = frame.shape[:2]
                    interpolation = cv2.INTER_AREA if (w > self.width or h > self.height) else cv2.INTER_LINEAR
                    resized = cv2.resize(frame, (self.width, self.height), interpolation=interpolation)
                    resized_frames.append(resized)
        else:
            # New optimization: batch resize frames with numpy
            # This is much faster for same-sized frames
            if len(frames) > 1 and all(f.shape == frames[0].shape for f in frames):
                try:
                    h, w = frames[0].shape[:2]
                    interpolation = cv2.INTER_AREA if (w > self.width or h > self.height) else cv2.INTER_LINEAR
                    
                    # Stack frames into a single array - much faster to process at once
                    if len(frames) > 4 and pre_resized:
                        # For pre-resized large batches, split into smaller chunks to avoid memory issues
                        chunk_size = 4
                        resized_frames = []
                        
                        for i in range(0, len(frames), chunk_size):
                            chunk = frames[i:i+chunk_size]
                            frames_array = np.stack(chunk)
                            resized_list = []
                            
                            for j in range(len(chunk)):
                                resized = cv2.resize(frames_array[j], (self.width, self.height), 
                                                   interpolation=interpolation)
                                resized_list.append(resized)
                                
                            resized_frames.extend(resized_list)
                    else:
                        # Default case: regular batch processing
                        frames_array = np.stack(frames)
                        resized_frames = []
                        
                        for i in range(len(frames)):
                            resized = cv2.resize(frames_array[i], (self.width, self.height), 
                                               interpolation=interpolation)
                            resized_frames.append(resized)
                except Exception as e:
                    self.logger.error(f"Error in batch resizing: {e}, falling back to individual resize")
                    # Fall back to individual frame resizing on error
                    resized_frames = []
                    for frame in frames:
                        h, w = frame.shape[:2]
                        interpolation = cv2.INTER_AREA if (w > self.width or h > self.height) else cv2.INTER_LINEAR
                        resized = cv2.resize(frame, (self.width, self.height), interpolation=interpolation)
                        resized_frames.append(resized)
            else:
                # Use individual frame resizing for varied sizes
                resized_frames = []
                
                for frame in frames:
                    # Determine if we're upscaling or downscaling for optimal interpolation
                    h, w = frame.shape[:2]
                    if w > self.width or h > self.height:
                        # Downscaling - use INTER_AREA for better quality
                        resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                    else:
                        # Upscaling - use INTER_LINEAR for speed
                        resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                    resized_frames.append(resized)
        
        self.logger.info(f"Prepared {len(resized_frames)} frames for detection")
        return resized_frames
            
    def get_batch(self) -> Tuple[Optional[List[int]], Optional[List[np.ndarray]]]:
        """Get a batch of prepared frames."""
        if self.stop_event.is_set():
            return None, None
        
        # Try several times if the prefetch thread is still alive, as it might be preparing a batch
        attempts = 0
        max_attempts = 5
        
        while attempts < max_attempts:
            try:
                # Wait longer for the first attempt, less for subsequent attempts
                timeout = 5.0 if attempts == 0 else 1.0
                self.logger.info(f"Waiting for batch (attempt {attempts+1}/{max_attempts}, timeout={timeout}s)")
                
                # Add timeout to avoid blocking forever
                item = self.prefetch_queue.get(timeout=timeout)
                self.prefetch_queue.task_done()
                
                if item is None:
                    self.logger.info("Received None from prefetch queue - end of processing")
                    return None, None
                
                self.logger.info(f"Got batch with {len(item[1])} frames")
                return item
            except queue.Empty:
                self.logger.info("Prefetch queue empty, checking if prefetch thread is still active")
                if not self.prefetch_thread.is_alive():
                    self.logger.info("Prefetch thread is no longer alive, no more batches expected")
                    return None, None
                
                # Increment attempt counter
                attempts += 1
                self.logger.info(f"No batch available yet, attempt {attempts}/{max_attempts}")
        
        self.logger.warning(f"No batch available after {max_attempts} attempts")
        return None, None
            
    def put_results(self, batch_indices: List[int], frames: List[np.ndarray], 
                  metadata: Optional[List[Dict]] = None):
        """Put processed results into result queue."""
        if self.stop_event.is_set():
            return
            
        try:
            self.result_queue.put((batch_indices, frames, metadata), timeout=1.0)
        except queue.Full:
            self.logger.warning("Result queue full, dropping batch results")
        except Exception as e:
            self.logger.error(f"Error putting results in queue: {str(e)}")
            
    def stop(self):
        """Stop all threads and clear queues."""
        # Signal threads to stop
        self.stop_event.set()
        self.stopped = True
        
        # Wait for threads to finish
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=2.0)
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=2.0)
            
        # Close video capture
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            
        # Clear queues
        self._clear_queue(self.frame_queue)
        self._clear_queue(self.prefetch_queue)
        self._clear_queue(self.result_queue)
        
        self.logger.debug("All threads stopped and queues cleared")
        
    def _clear_queue(self, q):
        """Safely clear a queue."""
        try:
            while True:
                q.get_nowait()
                q.task_done()
        except (queue.Empty, AttributeError):
            pass
            
    def get_statistics(self):
        """Get processing statistics."""
        return self.stats

class OptimizedVideoProcessor:
    """
    Optimized video processor that uses the frame buffer for efficient
    frame reading, processing, and saving with performance optimizations.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configure processing parameters
        self.batch_size = config['processing'].get('batch_size', 32)
        self.frame_skip = config['processing'].get('frame_skip', 5)
        
        # Multiprocessing configuration
        self.use_multiprocessing = config['processing'].get('use_multiprocessing', True)
        
        # Only use multiprocessing if batch size is large enough to overcome startup costs
        if self.batch_size < 16 and self.use_multiprocessing:
            self.logger.info(f"Batch size {self.batch_size} too small for efficient multiprocessing, disabling")
            self.use_multiprocessing = False
            
        if self.use_multiprocessing:
            cpu_count = os.cpu_count() or 4
            # Use fewer workers than cores to avoid thrashing
            self.resize_workers = min(cpu_count - 1, 4)
            self.resize_workers = max(2, self.resize_workers)  # At least 2 workers
            
            # Initialize multiprocessing pools if enabled
            if self.use_multiprocessing:
                try:
                    if ProcessPoolExecutor is None:
                        raise ImportError("ProcessPoolExecutor not available")
                    self.resize_pool = ProcessPoolExecutor(max_workers=self.resize_workers)
                    self.logger.info(f"Using ProcessPoolExecutor with {self.resize_workers} workers for frame resizing")
                except Exception as e:
                    self.logger.warning(f"ProcessPoolExecutor error: {e}, falling back to ThreadPoolExecutor")
                    self.resize_pool = ThreadPoolExecutor(max_workers=self.resize_workers)
        
        # We need to pass a reference to this processor to the frame buffer
        # for multiprocessing support
        config['parent'] = self
        
        # Initialize frame buffer (optimized)
        self.frame_buffer = OptimizedFrameBuffer(config)
        
        # Create a thread pool for asynchronous frame saving
        self.save_pool = ThreadPoolExecutor(max_workers=4)
        self.pending_saves = []
        self.pending_resizes = []
        
        # Set up output directories
        self.output_dir = Path(config['output'].get('directory', 'output'))
        self.raw_dir = self.output_dir / 'raw'
        self.processed_dir = self.output_dir / 'processed'
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Performance metrics
        self.perf_metrics = {
            "total_time": 0,
            "read_time": 0,
            "resize_time": 0,
            "detection_time": 0,
            "save_time": 0,
            "frames_processed": 0,
            "frames_saved": 0
        }
        
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'frame_buffer'):
            self.frame_buffer.stop()
            
        if hasattr(self, 'save_pool'):
            # Wait for pending saves to complete
            for future in self.pending_saves:
                try:
                    future.result(timeout=1.0)
                except:
                    pass
            
            self.save_pool.shutdown(wait=True)
            
        if hasattr(self, 'resize_pool') and self.use_multiprocessing:
            self.resize_pool.shutdown(wait=True)
            
        # Print performance metrics on cleanup
        self._print_performance_metrics()
            
    def process_video(self, video_path: str):
        """Process a single video file."""
        self.logger.info(f"Processing video: {video_path}")
        
        try:
            # Reset frame buffer for new video
            self.frame_buffer.reset()
            
            # Extract timestamp from filename (optional)
            video_name = os.path.basename(video_path)
            timestamp = self._extract_timestamp(video_name)
            
            # Start processing
            processing_start = time.time()
            
            # Start frame buffer threads
            if not self.frame_buffer.start_read_thread(video_path):
                self.logger.error(f"Failed to start frame reading for {video_path}")
                return False
                
            self.frame_buffer.start_prefetch_thread()
            
            # Process batches
            frames_processed = 0
            frames_saved = 0
            
            # Add debugging to track frame processing
            batch_count = 0
            self.logger.info("Starting batch processing loop")
            
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
                
                # Log progress
                progress = min(100, int(frames_processed / self.frame_buffer.frames_to_process * 100))
                elapsed = time.time() - processing_start
                eta = (elapsed / frames_processed) * (self.frame_buffer.frames_to_process - frames_processed) if frames_processed > 0 else 0
                
                self.logger.info(f"Progress: {progress}% ({frames_processed}/{self.frame_buffer.frames_to_process} frames), "
                                f"Saved: {frames_saved} frames, ETA: {eta:.1f}s")
                
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
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
            
        finally:
            # Clean up resources for this video
            self.frame_buffer.stop()
            
    def _process_batch(self, batch_indices: List[int], batch_frames: List[np.ndarray]):
        """Process a batch of frames and save results."""
        # Debug logging
        self.logger.info(f"Processing batch with {len(batch_frames)} frames")
        
        # Import here to avoid circular imports
        try:
            # Fix import path - use correct import path based on directory structure
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from src.stage1.person_detector import PersonDetector
            self.logger.info("Successfully imported PersonDetector")
        except ImportError as e:
            self.logger.error(f"Error importing PersonDetector: {e}")
            # Log current directory and sys.path for debugging
            self.logger.error(f"Current directory: {os.getcwd()}")
            self.logger.error(f"Python path: {sys.path}")
            # Return empty results if we can't import the detector
            return [None] * len(batch_frames)
        
        # Lazy load the detector on first use
        if not hasattr(self, 'detector'):
            self.logger.info("Initializing PersonDetector")
            try:
                # Set warmup size based on batch size for quicker startup
                warmup_size = min(4, self.batch_size)
                # Pass warmup size to PersonDetector if it supports it
                self.config['processing']['warmup_size'] = warmup_size
                
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
                
            # Process valid frames
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
                save_operations.append((frame_copy, detections))
                
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
            for frame, detections in save_operations:
                future = self.save_pool.submit(
                    self._save_frame_and_metadata,
                    frame,
                    detections
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
        
    def _save_frame_and_metadata(self, frame: np.ndarray, detections: List[Dict]) -> None:
        """Save a frame and its metadata."""
        save_start = time.time()
        
        try:
            # Generate unique filename using current time
            frame_id = int(datetime.now().timestamp() * 1000)
            
            # Convert the frame back to BGR for saving with OpenCV
            # Note: If frames were converted to RGB in _prepare_batch, we need to convert back
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Use memory-efficient approach for high-resolution frames
            h, w = frame_bgr.shape[:2]
            # Apply additional compression for very large frames
            jpeg_quality = 90  # Default quality
            if w * h > 2000000:  # 2MP+ frames
                jpeg_quality = 75  # Use more compression for high-res images
            
            # Save frame (using jpg format with adaptive quality)
            frame_path = self.processed_dir / f"frame_{frame_id}.jpg"
            cv2.imwrite(str(frame_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            
            # Save metadata using more efficient approach
            metadata_path = self.processed_dir / f"frame_{frame_id}.json"
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "detections": detections
            }
            
            # Write metadata with minimal formatting for efficiency
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, separators=(',', ':'))  # Use compact JSON
                
            save_time = time.time() - save_start
            self.perf_metrics["save_time"] += save_time
        except Exception as e:
            self.logger.error(f"Error saving frame: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
    def _extract_timestamp(self, filename: str) -> Optional[int]:
        """Extract timestamp from filename."""
        try:
            # Assuming filename format like "00M44S_1728867644.mp4"
            parts = filename.split('_')
            if len(parts) >= 2:
                ts_part = parts[1].split('.')[0]
                return int(ts_part)
        except:
            pass
            
        return None
            
    def _print_performance_metrics(self):
        """Print performance metrics."""
        if self.perf_metrics["total_time"] == 0:
            return
            
        total_wall_time = self.perf_metrics["total_time"]
        total_thread_time = (
            self.perf_metrics["read_time"] +
            self.perf_metrics["resize_time"] +
            self.perf_metrics["detection_time"] +
            self.perf_metrics["save_time"]
        )
        processed = self.perf_metrics["frames_processed"]
        saved = self.perf_metrics["frames_saved"]
        
        # Print to console
        print("\n" + "=" * 50)
        print("Performance Metrics:")
        print("-" * 50)
        print(f"Wall clock time: {total_wall_time:.2f}s")
        print(f"Total thread time: {total_thread_time:.2f}s")
        print(f"Frames processed: {processed}")
        print(f"Frames saved: {saved}")
        
        # Also log to file via logger
        self.logger.warning("=" * 50)
        self.logger.warning("Performance Metrics:")
        self.logger.warning("-" * 50)
        self.logger.warning(f"Wall clock time: {total_wall_time:.2f}s")
        self.logger.warning(f"Total thread time: {total_thread_time:.2f}s")
        self.logger.warning(f"Frames processed: {processed}")
        self.logger.warning(f"Frames saved: {saved}")
        
        if processed > 0:
            fps = processed / total_wall_time
            print(f"Average FPS: {fps:.2f}")
            self.logger.warning(f"Average FPS: {fps:.2f}")
            
            # Component percentages based on total thread time
            read_pct = (self.perf_metrics["read_time"] / total_thread_time) * 100
            resize_pct = (self.perf_metrics["resize_time"] / total_thread_time) * 100
            detect_pct = (self.perf_metrics["detection_time"] / total_thread_time) * 100
            save_pct = (self.perf_metrics["save_time"] / total_thread_time) * 100
            
            print(f"Time breakdown (% of total thread time):")
            print(f"Read time: {self.perf_metrics['read_time']:.2f}s ({read_pct:.1f}%)")
            print(f"Resize time: {self.perf_metrics['resize_time']:.2f}s ({resize_pct:.1f}%)")
            print(f"Detection time: {self.perf_metrics['detection_time']:.2f}s ({detect_pct:.1f}%)")
            print(f"Save time: {self.perf_metrics['save_time']:.2f}s ({save_pct:.1f}%)")
            
            self.logger.warning(f"Time breakdown (% of total thread time):")
            self.logger.warning(f"Read time: {self.perf_metrics['read_time']:.2f}s ({read_pct:.1f}%)")
            self.logger.warning(f"Resize time: {self.perf_metrics['resize_time']:.2f}s ({resize_pct:.1f}%)")
            self.logger.warning(f"Detection time: {self.perf_metrics['detection_time']:.2f}s ({detect_pct:.1f}%)")
            self.logger.warning(f"Save time: {self.perf_metrics['save_time']:.2f}s ({save_pct:.1f}%)")
            
            # Efficiency from parallelization
            if total_thread_time > 0:
                efficiency = (total_thread_time / total_wall_time) * 100
                print(f"Parallelization efficiency: {efficiency:.1f}% (higher is better)")
                print("Note: Values exceeding 100% indicate effective use of parallelization")
                
                self.logger.warning(f"Parallelization efficiency: {efficiency:.1f}% (higher is better)")
                self.logger.warning("Note: Values exceeding 100% indicate effective use of parallelization")
            
        print("=" * 50)
        self.logger.warning("=" * 50)
        
def main():
    """Main function to test the optimized processor."""
    # Get system information for optimal defaults
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
    
    parser = argparse.ArgumentParser(description="Optimized video processor for human detection")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--config", default="config/default_config.yaml", help="Path to the config file")
    parser.add_argument("--batch_size", type=int, default=min(32, max(8, cpu_count * 2)), 
                       help="Batch size for processing")
    parser.add_argument("--frame_skip", type=int, default=5,
                       help="Number of frames to skip (0 means no skip)")
    parser.add_argument("--output_dir", default=None, help="Custom output directory")
    parser.add_argument("--frame_queue_size", type=int, default=min(128, int(memory_gb * 10)),
                       help="Frame queue size")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--workers", type=int, default=min(4, max(2, cpu_count - 1)),
                       help="Number of worker processes for parallel operations")
    parser.add_argument("--confidence", type=float, default=0.3,
                       help="Detection confidence threshold (0.0-1.0)")
    parser.add_argument("--person_count", type=int, default=1,
                       help="Minimum person count required in a frame")
    
    args = parser.parse_args()
    
    # Configure logging
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("optimized_processor")
    
    # Log system and configuration info
    logger.info(f"System: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
    logger.info(f"Starting optimized processor with batch size {args.batch_size}, "
               f"frame skip {args.frame_skip}, workers {args.workers}, "
               f"confidence {args.confidence}, min_persons {args.person_count}")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override configuration with command line arguments
    if args.frame_skip:
        config['processing']['frame_skip'] = args.frame_skip
    
    if args.output_dir:
        config['output']['processed_dir'] = args.output_dir
    
    # Override detection settings for better results
    config['detection']['confidence_threshold'] = args.confidence
    config['detection']['person_count'] = args.person_count
    
    config['processing']['batch_size'] = args.batch_size
    config['processing']['workers'] = args.workers
    config['processing']['frame_queue_size'] = args.frame_queue_size
    
    # Process video
    processor = OptimizedVideoProcessor(config)
    try:
        processor.process_video(args.video)
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Make sure to shut down process pools
        logger.info("Shutting down processor")
        if hasattr(processor, 'save_pool'):
            processor.save_pool.shutdown(wait=False)
        logger.info("Processing completed")

if __name__ == "__main__":
    main() 