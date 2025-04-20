"""
Video Logs Extract - Main Entry Point

This script processes video files through multiple stages:
- Stage 0: Filter videos based on date and time constraints
- Stage 1: Initial frame extraction with person detection
- Future stages: Additional processing steps
"""

import sys
import logging
from pathlib import Path
import yaml
from dotenv import load_dotenv
import os
import argparse
import json

from src.stage1 import OptimizedVideoProcessor as Stage1Processor

def load_env_config():
    """Load configuration from environment variables."""
    load_dotenv()
    
    # Required settings
    input_dir = os.getenv('INPUT_DIR')
    if not input_dir:
        print("Error: INPUT_DIR environment variable is required")
        sys.exit(1)
        
    output_dir = os.getenv('OUTPUT_DIR')
    if not output_dir:
        print("Error: OUTPUT_DIR environment variable is required")
        sys.exit(1)
        
    # Optional settings with defaults
    return {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'confidence': os.getenv('CONFIDENCE'),
        'skip_frames': os.getenv('SKIP_FRAMES'),
        'format': os.getenv('FORMAT'),
        'config_file': os.getenv('CONFIG_FILE', 'config/default_config.yaml'),
        'stage': os.getenv('STAGE', 'stage1'),
        'resume': os.getenv('RESUME', 'true').lower() in ('true', 'yes', '1'),
        'reset': os.getenv('RESET', 'false').lower() in ('true', 'yes', '1')
    }

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)

def setup_logging(level: str = 'INFO'):
    """Configure logging settings."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set stage1 logger to WARNING to reduce verbose output
    stage1_logger = logging.getLogger('src.stage1')
    stage1_logger.setLevel(logging.WARNING)
    
    # Also set optimized_processor logger to WARNING
    optimized_processor_logger = logging.getLogger('src.optimized_processor')
    optimized_processor_logger.setLevel(logging.WARNING)

def load_video_list(file_path: Path) -> list:
    """Load list of videos from stage0 output file."""
    try:
        with open(file_path) as f:
            data = json.load(f)
            return data.get('videos', [])
    except Exception as e:
        logging.error(f"Error loading video list from {file_path}: {e}")
        return []

def reset_resume_state():
    """Reset the resume state by deleting the stage1_output.json file."""
    resume_file = Path("data/stage1_output.json")
    if resume_file.exists():
        try:
            resume_file.unlink()
            print(f"Reset resume state by deleting {resume_file}")
        except Exception as e:
            logging.error(f"Error deleting resume file: {e}")

def process_stage1(config: dict, input_dir: Path, output_dir: Path, should_reset: bool = False):
    """Run stage 1 processing: Extract frames with person detection."""
    # Reset resume state if requested
    if should_reset:
        reset_resume_state()
    
    processor = None
    try:
        processor = Stage1Processor(config)
        
        # Always try to use stage0 filtered list first
        stage0_output = Path(config.get('stage0', {}).get('output_file', 'data/stage0_output.json'))
        if stage0_output.exists():
            logging.info(f"Using filtered video list from {stage0_output}")
            video_list = load_video_list(stage0_output)
            if video_list:
                # Check resume state - how many videos already processed
                resume_file = Path("data/stage1_output.json")
                processed_count = 0
                if resume_file.exists():
                    try:
                        with open(resume_file) as f:
                            data = json.load(f)
                            processed_count = data.get('count', 0)
                    except Exception:
                        pass
                
                logging.info(f"Processing {len(video_list)} videos from stage0 filter ({processed_count} already processed)")
                for video_path in video_list:
                    try:
                        processor.process_video(video_path)
                    except KeyboardInterrupt:
                        logging.warning("Processing interrupted by user")
                        raise
                    except Exception as e:
                        logging.error(f"Error processing video {video_path}: {e}")
                return
            else:
                logging.warning("No videos found in stage0 filter, falling back to directory processing")
        else:
            logging.warning(f"Stage0 output file {stage0_output} not found, falling back to directory processing")
                
        # Fallback to processing the entire directory
        logging.info(f"Processing all videos in directory: {input_dir}")
        processor.process_directory(str(input_dir))
    except KeyboardInterrupt:
        logging.warning("Processing interrupted by user")
        # Print performance metrics for work completed so far
        if processor:
            processor._print_performance_metrics()
        raise
    except Exception as e:
        logging.error(f"Error in stage 1 processing: {e}")
        # Print performance metrics even if there's an error
        if processor:
            processor._print_performance_metrics()
        raise
    finally:
        # Ensure resources are properly cleaned up
        if processor and hasattr(processor, 'frame_buffer'):
            try:
                processor.frame_buffer.stop()
            except:
                pass

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process video files through multiple stages.')
    parser.add_argument('stage', choices=['stage1'], help='Processing stage to run')
    parser.add_argument('--reset', action='store_true', help='Reset resume state and process all videos again')
    parser.add_argument('--no-resume', action='store_true', help='Disable resume (process all videos regardless of resume state)')
    args = parser.parse_args()
    
    # Load environment configuration
    env_config = load_env_config()
    
    # Override environment config with command line args
    if args.reset:
        env_config['reset'] = True
    if args.no_resume:
        env_config['resume'] = False
    
    # Setup paths
    input_dir = Path(env_config['input_dir'])
    output_dir = Path(env_config['output_dir'])
    config_path = Path(env_config['config_file'])
    
    # Validate paths
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)
    
    # Override config with environment variables
    if env_config['confidence']:
        config['detection']['confidence_threshold'] = float(env_config['confidence'])
    if env_config['skip_frames']:
        config['detection']['skip_frames'] = int(env_config['skip_frames'])
    if env_config['format']:
        config['output']['format'] = env_config['format']
    
    # Set resume option in config
    config['resume'] = env_config['resume']
            
    # Set output directory in config
    config['output']['directory'] = str(output_dir)
    
    # Setup logging
    setup_logging(config['processing'].get('log_level', 'INFO'))
    
    try:
        # Show resume status
        resume_state = "enabled" if config['resume'] else "disabled"
        reset_state = "yes" if env_config['reset'] else "no"
        print(f"Resume is {resume_state}, Reset: {reset_state}")
        
        # Run the specified stage
        if args.stage == 'stage1':
            process_stage1(config, input_dir, output_dir, env_config['reset'])
        else:
            print(f"Error: Stage '{args.stage}' not implemented")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 