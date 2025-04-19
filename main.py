#!/usr/bin/env python3

import sys
from pathlib import Path
import yaml
import logging
import os
from dotenv import load_dotenv
from src.video_processor import VideoProcessor
from src.utils import setup_logging

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_env_config():
    """Load and validate environment configuration."""
    # Load .env file if it exists
    load_dotenv()
    
    # Required settings
    input_dir = os.getenv('INPUT_DIR')
    output_dir = os.getenv('OUTPUT_DIR')
    
    if not input_dir or not output_dir:
        print("Error: INPUT_DIR and OUTPUT_DIR must be set in .env file")
        print("Please copy .env.example to .env and configure your settings")
        sys.exit(1)
    
    # Optional settings with their environment variable names
    return {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'config_file': os.getenv('CONFIG_FILE', 'config/default_config.yaml'),
        'confidence': os.getenv('CONFIDENCE'),
        'skip_frames': os.getenv('SKIP_FRAMES'),
        'format': os.getenv('OUTPUT_FORMAT'),
        'person_count': os.getenv('PERSON_COUNT')
    }

def main():
    """Main entry point."""
    # Load environment configuration
    env_config = load_env_config()
    
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
    if env_config['person_count']:
        config['detection']['person_count'] = int(env_config['person_count'])
        
    # Set output directory in config
    config['output']['directory'] = str(output_dir)
    
    # Setup logging
    setup_logging(config['processing'].get('log_level', 'INFO'))
    
    try:
        # Initialize video processor
        processor = VideoProcessor(config)
        
        # Process all videos in input directory
        processor.process_directory(str(input_dir))
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 