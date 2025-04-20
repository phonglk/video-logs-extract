import sys
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from .video_collector import VideoCollector

def main():
    # Load environment variables from .env
    load_dotenv()
    
    # Get input directory from environment variable or command line
    input_dir = os.getenv('INPUT_DIR')
    
    # Fallback to command line argument if INPUT_DIR not set
    if not input_dir:
        if len(sys.argv) != 2:
            print("Error: INPUT_DIR environment variable not set")
            print("Usage: INPUT_DIR=/path/to/videos python -m src.stage0.main")
            print("   or: python -m src.stage0.main <input_directory>")
            sys.exit(1)
        input_dir = sys.argv[1]
    
    # Validate input directory
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Load config
    config_path = Path(os.getenv('CONFIG_FILE', 'config/default_config.yaml'))
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
        
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Get output file path from config
    script_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    output_file = script_dir / config.get('stage0', {}).get('output_file', 'data/stage0_output.json')
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
        
    # Initialize collector
    collector = VideoCollector(config)
    
    # Collect and filter videos
    print("\nConfiguration:")
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Start date filter: {config['processing'].get('start_date')}")
    print(f"Time ranges: {config['processing'].get('time_ranges')}")
    print("\nAnalyzing videos...\n")
    
    videos = collector.collect_videos(input_dir)
    
    if videos:
        print(f"\nFound {len(videos)} videos to process:")
        for i, video in enumerate(videos, 1):
            print(f"{i:3d}. {video.relative_to(input_dir)}")
            
        # Save to file
        collector.save_to_file(videos, output_file)
        print(f"\nVideo list saved to: {output_file}")
    else:
        print("\nNo videos found matching the criteria.")
        sys.exit(1)
        
if __name__ == "__main__":
    main() 