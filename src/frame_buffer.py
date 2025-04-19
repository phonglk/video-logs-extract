from queue import Queue, Empty
from threading import Thread, Event
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import time
import logging

class FrameBuffer:
    def __init__(self, config: Dict[str, Any]):
        """Initialize frame buffer with configuration."""
        self.config = config
        self.reset()
        
    def reset(self):
        """Reset buffer state for new video."""
        # Stop existing threads if any
        if hasattr(self, 'stop_event'):
            self.stop()
            
        # Create new queues and event
        self.frame_queue = Queue(maxsize=self.config['processing'].get('frame_queue_size', 512))
        self.result_queue = Queue(maxsize=self.config['processing'].get('result_queue_size', 128))
        self.prefetch_queue = Queue(maxsize=self.config['processing'].get('prefetch_batches', 4))
        self.stop_event = Event()
        self.logger = logging.getLogger(__name__)
        
    def start_read_thread(self, cap: cv2.VideoCapture, skip_frames: int = 0):
        """Start thread for reading frames from video."""
        # Reset buffer state for new video
        self.reset()
        
        def read_frames():
            frame_count = 0
            batch = []
            try:
                while not self.stop_event.is_set():
                    # Read a frame
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame_count += 1
                    
                    # Add frame to batch
                    batch.append((frame, frame_count))
                    
                    # Skip frames if needed
                    if skip_frames > 0:
                        for _ in range(skip_frames):
                            ret, _ = cap.read()
                            # Even if read fails, we count it as processed
                            frame_count += 1
                            if not ret:
                                # Don't break, just stop skipping
                                break
                    
                    # When batch is full, add to frame queue
                    if len(batch) >= self.config['processing'].get('batch_size', 64):
                        if not self.stop_event.is_set():  # Check again before putting
                            self.frame_queue.put(batch)
                            batch = []
                        
            except Exception as e:
                self.logger.error(f"Error in read thread: {e}")
            finally:
                # Put remaining frames
                if batch and not self.stop_event.is_set():
                    self.frame_queue.put(batch)
                # Signal end of video
                try:
                    self.frame_queue.put(None, timeout=1.0)
                except:
                    pass
                    
        self.read_thread = Thread(target=read_frames, daemon=True)
        self.read_thread.start()
        
    def start_prefetch_thread(self):
        """Start thread for prefetching and preparing batches."""
        def prefetch_frames():
            while not self.stop_event.is_set():
                try:
                    batch = self.frame_queue.get(timeout=1.0)
                    if batch is None:
                        self.prefetch_queue.put(None)
                        break
                        
                    # Prepare frames in batch (resize etc.)
                    prepared_batch = []
                    for frame, frame_num in batch:
                        # Add frame preparation here if needed
                        prepared_batch.append((frame, frame_num))
                        
                    self.prefetch_queue.put(prepared_batch)
                    self.frame_queue.task_done()
                except Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in prefetch thread: {e}")
                    continue
                    
        self.prefetch_thread = Thread(target=prefetch_frames, daemon=True)
        self.prefetch_thread.start()
        
    def get_batch(self) -> Tuple[List[np.ndarray], List[int]]:
        """Get a prepared batch of frames for processing."""
        try:
            batch = self.prefetch_queue.get(timeout=5.0)
            if batch is None:
                return [], []
                
            frames, frame_numbers = zip(*batch)
            self.prefetch_queue.task_done()
            return list(frames), list(frame_numbers)
            
        except Empty:
            return [], []
        except Exception as e:
            self.logger.error(f"Error getting batch: {e}")
            return [], []
        
    def put_results(self, frames: List[np.ndarray], metadata_list: List[dict]):
        """Put processed results into result queue."""
        try:
            for frame, metadata in zip(frames, metadata_list):
                self.result_queue.put((frame, metadata))
        except Exception as e:
            self.logger.error(f"Error putting results: {e}")
            
    def stop(self):
        """Stop all threads and clear queues."""
        if hasattr(self, 'stop_event'):
            self.stop_event.set()
        
        # Clear queues with timeout to prevent hanging
        try:
            while True:
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    break
                    
            while True:
                try:
                    self.prefetch_queue.get_nowait()
                except Empty:
                    break
                    
            while True:
                try:
                    self.result_queue.get_nowait()
                except Empty:
                    break
        except:
            pass
            
        # Wait for threads to finish with timeout
        if hasattr(self, 'read_thread'):
            self.read_thread.join(timeout=2.0)
        if hasattr(self, 'prefetch_thread'):
            self.prefetch_thread.join(timeout=2.0)
            
        # Mark all tasks as done to prevent hanging
        try:
            self.frame_queue.task_done()
            self.prefetch_queue.task_done()
            self.result_queue.task_done()
        except:
            pass
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop() 