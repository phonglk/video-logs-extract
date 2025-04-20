#!/usr/bin/env python
"""
Run the optimized video processor on a video file or directory.
This script provides a command-line interface to the OptimizedVideoProcessor.
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
import psutil
import multiprocessing as mp

# Add the src directory to the Python path
src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

from src.stage1 import OptimizedVideoProcessor

def main():
    # Get system information for optimal defaults
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
    
    parser = argparse.ArgumentParser(description='Run the optimized video processor')
    parser.add_argument('--config', default='config/default_config.yaml', help='Path to the configuration file')
    parser.add_argument('--input', required=True, help='Path to video file or directory of videos')
    parser.add_argument('--output', help='Output directory (overrides config)')
    parser.add_argument('--batch-size', type=int, default=min(32, max(8, cpu_count * 2)), 
                      help='Batch size for processing')
    parser.add_argument('--frame-skip', type=int, default=5, 
                      help='Number of frames to skip (0 means no skip)')
    parser.add_argument('--confidence', type=float, default=0.3,
                      help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--person-count', type=int, default=1,
                      help='Minimum person count required in a frame')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--no-resume', action='store_true', help='Disable resume support')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('optimized_processor')
    
    # Configure logging for different modules
    if args.debug:
        # In debug mode, enable all logs at DEBUG level
        # Set all relevant loggers to DEBUG
        logging.getLogger('src').setLevel(logging.DEBUG)
        logging.getLogger('src.stage1').setLevel(logging.DEBUG)
        logging.getLogger('src.optimized_processor').setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled - showing all logs")
    else:
        # In normal mode, limit verbose logs
        stage1_logger = logging.getLogger('src.stage1')
        stage1_logger.setLevel(logging.WARNING)
        
        # Also set optimized_processor logger to WARNING
        optimized_processor_logger = logging.getLogger('src.optimized_processor')
        optimized_processor_logger.setLevel(logging.WARNING)
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        return 1
    
    # Ensure necessary config sections exist
    if 'processing' not in config:
        config['processing'] = {}
    if 'detection' not in config:
        config['detection'] = {}
    if 'output' not in config:
        config['output'] = {}
    
    # Override configuration with command line arguments
    config['processing']['batch_size'] = args.batch_size
    config['processing']['frame_skip'] = args.frame_skip
    config['detection']['confidence_threshold'] = args.confidence
    config['detection']['person_count'] = args.person_count
    config['processing']['log_level'] = 'DEBUG' if args.debug else 'INFO'
    
    if args.output:
        config['output']['directory'] = args.output
    elif 'directory' not in config['output']:
        # Set default output directory if not specified
        config['output']['directory'] = 'output'
    
    # Disable resume if requested
    if args.no_resume:
        config['resume'] = False
    
    # Log configuration
    logger.info(f"Starting optimized processor with config:")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Frame skip: {args.frame_skip}")
    logger.info(f"  Confidence: {args.confidence}")
    logger.info(f"  Person count: {args.person_count}")
    logger.info(f"  Input: {args.input}")
    logger.info(f"  Output: {config['output']['directory']}")
    
    # Create the processor
    processor = OptimizedVideoProcessor(config)
    
    # Process the input with proper error handling and cancellation support
    try:
        input_path = Path(args.input)
        if input_path.is_file():
            # Process a single video file
            processor.process_video(str(input_path))
        elif input_path.is_dir():
            # Process a directory of videos
            processor.process_directory(str(input_path))
        else:
            logger.error(f"Input path does not exist: {input_path}")
            return 1
        
        # Always show performance metrics when processing completes successfully
        # No need to call _print_performance_metrics() here as it will be called in __del__
        
        logger.info("Processing complete")
        return 0
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        # Print statistics for completed work
        processor._print_performance_metrics()
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Print statistics for completed work even when an error occurs
        processor._print_performance_metrics()
        return 1
    finally:
        # Ensure resources are cleaned up
        try:
            # Clean up resources - this will also print metrics through __del__
            processor.frame_buffer.stop()
        except:
            pass

if __name__ == '__main__':
    sys.exit(main()) 