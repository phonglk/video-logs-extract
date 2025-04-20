#!/usr/bin/env python
import time
import argparse
import cv2
import numpy as np
import os
import yaml
import logging
import torch
import psutil
import multiprocessing
from pathlib import Path
from typing import List, Dict, Any, Tuple
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("profiler")

def get_cpu_usage():
    """Get current CPU utilization per core."""
    return psutil.cpu_percent(interval=0.1, percpu=True)

def log_system_info():
    """Log system information including CPU cores and memory."""
    logger.info(f"CPU Cores: {psutil.cpu_count(logical=False)} Physical, {psutil.cpu_count()} Logical")
    logger.info(f"Available Memory: {psutil.virtual_memory().available / (1024 * 1024 * 1024):.2f} GB")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available'):
        logger.info(f"MPS available: {torch.backends.mps.is_available()}")

def load_config(config_path: str = "config/default_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def profile_video_read(video_path: str, batch_size: int, frame_skip: int, num_frames: int = -1) -> Dict[str, float]:
    """
    Profile reading frames from a video.
    
    Args:
        video_path: Path to the video file
        batch_size: Number of frames to read in each batch
        frame_skip: Number of frames to skip between reads
        num_frames: Maximum number of frames to read (-1 for all)
        
    Returns:
        Dictionary with profiling results
    """
    results = {
        "total_time": 0,
        "frames_read": 0,
        "batches_read": 0,
        "avg_time_per_frame": 0,
        "avg_time_per_batch": 0,
        "fps": 0
    }
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return results
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video: {os.path.basename(video_path)}")
    logger.info(f"Total frames: {total_frames}, FPS: {video_fps}, Resolution: {width}x{height}")
    
    if num_frames > 0:
        max_frames = min(num_frames, total_frames)
    else:
        max_frames = total_frames
    
    logger.info(f"Reading {max_frames} frames with batch_size={batch_size}, frame_skip={frame_skip}")
    
    start_time = time.time()
    frame_count = 0
    batch_count = 0
    
    try:
        while frame_count < max_frames:
            batch = []
            for _ in range(batch_size):
                if frame_count >= max_frames:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                batch.append(frame)
                frame_count += 1
                
                # Skip frames if requested
                for _ in range(frame_skip):
                    if frame_count >= max_frames:
                        break
                    cap.read()  # Discard the frame
                    frame_count += 1
            
            if not batch:
                break
                
            batch_count += 1
            
            if batch_count % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Progress: {frame_count}/{max_frames} frames, {elapsed:.2f}s elapsed, "
                            f"{frame_count/elapsed:.2f} fps")
    
    except Exception as e:
        logger.error(f"Error during video reading: {e}")
    finally:
        cap.release()
        
    total_time = time.time() - start_time
    results["total_time"] = total_time
    results["frames_read"] = frame_count
    results["batches_read"] = batch_count
    
    if frame_count > 0:
        results["avg_time_per_frame"] = total_time / frame_count
        results["fps"] = frame_count / total_time
    
    if batch_count > 0:
        results["avg_time_per_batch"] = total_time / batch_count
    
    return results

def profile_frame_resize(video_path: str, target_size: Tuple[int, int], batch_size: int = 32, 
                        frame_skip: int = 5, num_frames: int = 1000, multicore: bool = False) -> Dict[str, float]:
    """
    Profile frame resizing operations.
    
    Args:
        video_path: Path to the video file
        target_size: Target size for resizing (width, height)
        batch_size: Number of frames to process in each batch
        frame_skip: Number of frames to skip between reads
        num_frames: Maximum number of frames to process
        multicore: Use multiprocessing for parallel resizing
        
    Returns:
        Dictionary with profiling results
    """
    results = {
        "total_time": 0,
        "resize_time": 0,
        "frames_processed": 0,
        "batches_processed": 0,
        "avg_time_per_frame": 0,
        "fps": 0,
        "cpu_usage": []
    }
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return results
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames > 0:
        max_frames = min(num_frames, total_frames)
    else:
        max_frames = total_frames
    
    logger.info(f"Resizing {max_frames} frames to {target_size[0]}x{target_size[1]}" + 
               (" with multiprocessing" if multicore else ""))
    
    # Prepare for multiprocessing
    if multicore:
        # Use only 70% of available cores to avoid thrashing
        cpu_count = max(2, int(multiprocessing.cpu_count() * 0.7))
        pool = multiprocessing.Pool(processes=cpu_count)
        logger.info(f"Using multiprocessing pool with {cpu_count} processes")
    
    frame_count = 0
    batch_count = 0
    total_start_time = time.time()
    resize_time = 0
    
    # Run two tests: individual frame resize and batch resize
    individual_resize_time = 0
    batch_resize_time = 0
    
    # CPU monitoring setup
    cpu_monitor_interval = max(1, int(max_frames / 20))  # Sample CPU more frequently
    
    # Start CPU monitoring thread
    stop_monitor = threading.Event()
    cpu_samples = []
    
    def cpu_monitor():
        while not stop_monitor.is_set():
            cpu_samples.append(psutil.cpu_percent(interval=0.5, percpu=True))
            time.sleep(0.5)
    
    monitor_thread = threading.Thread(target=cpu_monitor, daemon=True)
    monitor_thread.start()
    
    try:
        while frame_count < max_frames:
            # Read batch of frames
            batch = []
            for _ in range(batch_size):
                if frame_count >= max_frames:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                batch.append(frame)
                frame_count += 1
                
                # Skip frames if requested
                for _ in range(frame_skip):
                    if frame_count >= max_frames:
                        break
                    cap.read()  # Discard the frame
                    frame_count += 1
            
            if not batch:
                break
                
            batch_count += 1
            
            # Measure individual resize time
            resize_start = time.time()
            
            if multicore:
                # Parallel resize using multiple processes
                resize_args = [(frame, target_size) for frame in batch]
                resized_frames = pool.starmap(parallel_resize, resize_args)
            else:
                # Single frame resize
                resized_frames = []
                for frame in batch:
                    resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
                    resized_frames.append(resized)
                    
            individual_time = time.time() - resize_start
            individual_resize_time += individual_time
            
            # Measure batch resize time
            resize_start = time.time()
            # Batch resize
            frames_array = np.stack(batch)
            resized_list = []
            for i in range(len(batch)):
                resized = cv2.resize(frames_array[i], target_size, interpolation=cv2.INTER_LINEAR)
                resized_list.append(resized)
            batch_time = time.time() - resize_start
            batch_resize_time += batch_time
            
            # Use the faster method for the results
            if individual_time <= batch_time:
                resize_time += individual_time
                method = "individual"
            else:
                resize_time += batch_time
                method = "batch"
                
            if batch_count % 10 == 0:
                elapsed = time.time() - total_start_time
                logger.info(f"Progress: {frame_count}/{max_frames} frames, {elapsed:.2f}s elapsed, "
                            f"{frame_count/elapsed:.2f} fps, {method} resize was faster")
                # Log current CPU usage
                if cpu_samples:
                    avg_usage = sum(cpu_samples[-1]) / len(cpu_samples[-1])
                    max_usage = max(cpu_samples[-1])
                    logger.info(f"CPU Usage: {avg_usage:.1f}% (avg), {max_usage:.1f}% (max)")
    
    except Exception as e:
        logger.error(f"Error during resize profiling: {e}")
    finally:
        # Stop CPU monitoring
        stop_monitor.set()
        monitor_thread.join(timeout=1.0)
        
        # Cleanup
        cap.release()
        if multicore:
            pool.close()
            pool.join()
        
    total_time = time.time() - total_start_time
    results["total_time"] = total_time
    results["resize_time"] = resize_time
    results["frames_processed"] = frame_count
    results["batches_processed"] = batch_count
    results["individual_resize_time"] = individual_resize_time
    results["batch_resize_time"] = batch_resize_time
    results["cpu_usage"] = cpu_samples
    
    if frame_count > 0:
        results["avg_time_per_frame"] = resize_time / frame_count
        results["resize_fps"] = frame_count / resize_time
        results["total_fps"] = frame_count / total_time
        results["individual_fps"] = frame_count / individual_resize_time if individual_resize_time > 0 else 0
        results["batch_fps"] = frame_count / batch_resize_time if batch_resize_time > 0 else 0
        
        # Calculate average CPU utilization
        if cpu_samples:
            all_cores = []
            for sample in cpu_samples:
                all_cores.extend(sample)
            results["avg_cpu_usage"] = sum(all_cores) / len(all_cores) if all_cores else 0
            results["max_cpu_usage"] = max(all_cores) if all_cores else 0
            
            if len(cpu_samples) > 0 and len(cpu_samples[0]) > 0:
                # Transpose to get per-core stats
                core_samples = list(zip(*cpu_samples))
                core_avgs = [sum(core) / len(core) for core in core_samples]
                results["core_usage"] = core_avgs
    
    return results

def parallel_resize(frame, target_size):
    """Resize a frame (for parallel processing)."""
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

def profile_detection_model(config: Dict[str, Any], video_path: str, batch_size: int = 32,
                          frame_skip: int = 5, num_frames: int = 1000) -> Dict[str, float]:
    """
    Profile the object detection model performance.
    
    Args:
        config: Configuration dictionary
        video_path: Path to the video file
        batch_size: Number of frames to process in each batch
        frame_skip: Number of frames to skip between reads
        num_frames: Maximum number of frames to process
        
    Returns:
        Dictionary with profiling results
    """
    try:
        # Import person detector
        from stage1.person_detector import PersonDetector
    except ImportError:
        logger.error("Could not import PersonDetector. Make sure you're in the correct directory.")
        return {}
        
    results = {
        "total_time": 0,
        "detection_time": 0,
        "frames_processed": 0,
        "batches_processed": 0,
        "avg_time_per_frame": 0,
        "avg_time_per_batch": 0,
        "fps": 0
    }
    
    # Initialize detection pipeline
    try:
        detector = PersonDetector(config)
    except Exception as e:
        logger.error(f"Error initializing PersonDetector: {e}")
        return results
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return results
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames > 0:
        max_frames = min(num_frames, total_frames)
    else:
        max_frames = total_frames
        
    target_size = (config['processing'].get('width', 640), config['processing'].get('height', 640))
    logger.info(f"Running detection on {max_frames} frames")
    
    frame_count = 0
    batch_count = 0
    total_start_time = time.time()
    detection_time = 0
    
    try:
        while frame_count < max_frames:
            # Read batch of frames
            batch = []
            for _ in range(batch_size):
                if frame_count >= max_frames:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame
                resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
                batch.append(resized)
                frame_count += 1
                
                # Skip frames if requested
                for _ in range(frame_skip):
                    if frame_count >= max_frames:
                        break
                    cap.read()  # Discard the frame
                    frame_count += 1
            
            if not batch:
                break
                
            batch_count += 1
            
            # Run detection
            detect_start = time.time()
            detections = detector.process_batch(batch)
            detection_time += time.time() - detect_start
            
            if batch_count % 5 == 0:
                elapsed = time.time() - total_start_time
                logger.info(f"Progress: {frame_count}/{max_frames} frames, {elapsed:.2f}s elapsed, "
                            f"{frame_count/elapsed:.2f} fps")
    
    except Exception as e:
        logger.error(f"Error during detection profiling: {e}")
    finally:
        cap.release()
        
    total_time = time.time() - total_start_time
    results["total_time"] = total_time
    results["detection_time"] = detection_time
    results["frames_processed"] = frame_count
    results["batches_processed"] = batch_count
    
    if frame_count > 0:
        results["avg_time_per_frame"] = detection_time / frame_count
        results["detection_fps"] = frame_count / detection_time
        results["total_fps"] = frame_count / total_time
    
    if batch_count > 0:
        results["avg_time_per_batch"] = detection_time / batch_count
    
    return results

def profile_save_operations(output_dir: str, num_frames: int = 100, 
                          frame_size: Tuple[int, int] = (640, 640)) -> Dict[str, float]:
    """
    Profile frame saving operations.
    
    Args:
        output_dir: Directory to save test frames
        num_frames: Number of frames to save
        frame_size: Size of test frames
        
    Returns:
        Dictionary with profiling results
    """
    results = {
        "total_time": 0,
        "frames_saved": 0,
        "avg_time_per_frame": 0,
        "fps": 0
    }
    
    # Test various image formats and compression levels
    formats = [
        {"name": "jpg_90", "ext": "jpg", "params": [cv2.IMWRITE_JPEG_QUALITY, 90]},
        {"name": "jpg_75", "ext": "jpg", "params": [cv2.IMWRITE_JPEG_QUALITY, 75]},
        {"name": "jpg_50", "ext": "jpg", "params": [cv2.IMWRITE_JPEG_QUALITY, 50]},
        {"name": "png", "ext": "png", "params": None},
        {"name": "png_compressed", "ext": "png", "params": [cv2.IMWRITE_PNG_COMPRESSION, 9]},
    ]
    
    # Create test frame
    test_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    
    # Draw something on the frame to make it compressible
    cv2.rectangle(test_frame, (50, 50), (frame_size[0] - 50, frame_size[1] - 50), (0, 255, 0), 2)
    cv2.putText(test_frame, "Test Frame", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.circle(test_frame, (frame_size[0] // 2, frame_size[1] // 2), 100, (0, 0, 255), 3)
    
    # Performance metrics for each format
    format_results = {}
    
    for fmt in formats:
        # Create test output directory for this format
        fmt_dir = os.path.join(output_dir, f"test_profile_{fmt['name']}")
        os.makedirs(fmt_dir, exist_ok=True)
        logger.info(f"Testing {fmt['name']} format: Saving {num_frames} frames to {fmt_dir}")
        
        # Measure time for this format
        start_time = time.time()
        total_size = 0
        
        for i in range(num_frames):
            frame_path = os.path.join(fmt_dir, f"frame_{i:04d}.{fmt['ext']}")
            metadata_path = os.path.join(fmt_dir, f"frame_{i:04d}.json")
            
            # Save frame with format-specific parameters
            if fmt['params']:
                cv2.imwrite(frame_path, test_frame, fmt['params'])
            else:
                cv2.imwrite(frame_path, test_frame)
                
            # Get file size
            total_size += os.path.getsize(frame_path)
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                f.write('{"timestamp": "2023-01-01T00:00:00", "detections": [{"class": "person", "confidence": 0.95, "box": [0.1, 0.1, 0.9, 0.9]}]}')
        
        fmt_time = time.time() - start_time
        avg_size = total_size / num_frames if num_frames > 0 else 0
        
        # Record metrics for this format
        format_results[fmt['name']] = {
            "time": fmt_time,
            "fps": num_frames / fmt_time if fmt_time > 0 else 0,
            "avg_file_size": avg_size,
            "total_size": total_size
        }
        
        logger.info(f"{fmt['name']}: {fmt_time:.4f}s, {format_results[fmt['name']]['fps']:.2f} fps, {avg_size/1024:.2f} KB avg")
    
    # Find the best format (highest fps)
    best_format = max(format_results.items(), key=lambda x: x[1]["fps"])
    logger.info(f"Best performing format: {best_format[0]} with {best_format[1]['fps']:.2f} fps")
    
    # Add to overall results
    results["total_time"] = sum(fmt["time"] for fmt in format_results.values())
    results["frames_saved"] = num_frames * len(formats)
    results["format_results"] = format_results
    
    if num_frames > 0:
        results["avg_time_per_frame"] = best_format[1]["time"] / num_frames
        results["fps"] = best_format[1]["fps"]
    
    return results

def print_results(title: str, results: Dict[str, float]):
    """Print formatted profiling results."""
    logger.info("=" * 60)
    logger.info(f"{title} Results:")
    logger.info("-" * 60)
    
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")
    
    logger.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Video Processing Profiler")
    parser.add_argument("--video", type=str, required=True, help="Path to video file for profiling")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="data/profile_output", help="Output directory for test files")
    parser.add_argument("--profile-type", type=str, choices=["read", "resize", "detect", "save", "all"], 
                      default="all", help="Which operation to profile")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--frame-skip", type=int, default=5, help="Frames to skip between processing")
    parser.add_argument("--num-frames", type=int, default=1000, help="Number of frames to process")
    parser.add_argument("--multicore", action="store_true", help="Use multiprocessing for CPU-bound operations")
    parser.add_argument("--compare", action="store_true", help="Compare single-core vs multi-core performance")
    
    args = parser.parse_args()
    
    # Ensure video exists
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set target size from config
    target_size = (config['processing'].get('width', 640), config['processing'].get('height', 640))
    
    # Log system information
    log_system_info()
        
    # Run selected profiling type(s)
    logger.info(f"Starting profiling for video: {args.video}")
    
    if args.profile_type in ["read", "all"]:
        results = profile_video_read(args.video, args.batch_size, args.frame_skip, args.num_frames)
        print_results("Video Reading", results)
        
    if args.profile_type in ["resize", "all"]:
        if args.compare:
            # Run without multiprocessing first
            logger.info("=== Single-core Resize Test ===")
            single_results = profile_frame_resize(
                args.video, target_size, args.batch_size, args.frame_skip, args.num_frames, multicore=False
            )
            print_results("Frame Resizing (Single-core)", single_results)
            
            # Run with multiprocessing
            logger.info("=== Multi-core Resize Test ===")
            multi_results = profile_frame_resize(
                args.video, target_size, args.batch_size, args.frame_skip, args.num_frames, multicore=True
            )
            print_results("Frame Resizing (Multi-core)", multi_results)
            
            # Print comparison
            speedup = multi_results["resize_fps"] / single_results["resize_fps"] if single_results["resize_fps"] > 0 else 0
            logger.info(f"Multi-core speedup: {speedup:.2f}x faster")
            logger.info(f"Single-core CPU usage: {single_results.get('avg_cpu_usage', 0):.1f}% avg, {single_results.get('max_cpu_usage', 0):.1f}% max")
            logger.info(f"Multi-core CPU usage: {multi_results.get('avg_cpu_usage', 0):.1f}% avg, {multi_results.get('max_cpu_usage', 0):.1f}% max")
        else:
            results = profile_frame_resize(
                args.video, target_size, args.batch_size, args.frame_skip, args.num_frames, multicore=args.multicore
            )
            print_results("Frame Resizing", results)
        
    if args.profile_type in ["detect", "all"]:
        results = profile_detection_model(config, args.video, args.batch_size, args.frame_skip, args.num_frames)
        print_results("Object Detection", results)
        
    if args.profile_type in ["save", "all"]:
        results = profile_save_operations(args.output_dir, num_frames=min(args.num_frames, 100))
        print_results("Frame Saving", results)
        
    logger.info("Profiling completed!")

if __name__ == "__main__":
    main() 