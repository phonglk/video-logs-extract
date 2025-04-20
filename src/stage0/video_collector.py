import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dateutil import parser
from dateutil.parser import ParserError

class VideoCollector:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the video collector with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configure logging
        logging.basicConfig(
            level=config['processing'].get('log_level', 'INFO'),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def save_to_file(self, videos: List[Path], output_file: Path) -> None:
        """Save the list of video paths to a JSON file.
        
        Args:
            videos: List of video Path objects
            output_file: Path to save the JSON file
        """
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert paths to strings and save to JSON
        video_data = {
            'total_videos': len(videos),
            'videos': [str(v) for v in videos],
            'generated_at': datetime.now().isoformat(),
            'config': {
                'start_date': self.config['processing'].get('start_date'),
                'time_ranges': self.config['processing'].get('time_ranges')
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(video_data, f, indent=2)
            
        self.logger.info(f"Saved {len(videos)} video paths to {output_file}")

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse a date string using dateutil's parser.
        
        Args:
            date_str: Date string in YYYYMMDD format
            
        Returns:
            datetime object or None if parsing fails
        """
        try:
            # First ensure it's in YYYYMMDD format by checking length and digits
            date_str = str(date_str)
            if not (len(date_str) == 8 and date_str.isdigit()):
                raise ValueError(f"Date string must be 8 digits (YYYYMMDD), got: {date_str}")
            
            # Format the string to be more parser-friendly
            formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            return parser.parse(formatted)
        except (ValueError, ParserError) as e:
            self.logger.error(f"Failed to parse date '{date_str}': {e}")
            return None

    def collect_videos(self, input_dir: str) -> List[Path]:
        """Collect and filter video files based on date and time constraints.
        
        Args:
            input_dir: Directory to scan for videos
            
        Returns:
            List of Path objects for videos that meet the constraints
        """
        input_path = Path(input_dir)
        supported_formats = self.config['input'].get(
            'supported_formats',
            ['.mp4', '.avi', '.mov']
        )
        
        # Get start date from config and parse
        start_date = self.config['processing'].get('start_date', '')
        if start_date:
            start_date_dt = self._parse_date(str(start_date))
            if not start_date_dt:
                self.logger.error("Invalid start date in config")
                return []
            self.logger.info(f"Start date filter: {start_date_dt.strftime('%Y-%m-%d')}")
        else:
            self.logger.warning("No start_date configured in config!")
            return []
            
        # First pass: collect all video files
        all_videos = []
        for fmt in supported_formats:
            all_videos.extend([
                f for f in input_path.rglob(f'*{fmt}')
                if not f.name.startswith('.')
            ])
            
        self.logger.info(f"Found {len(all_videos)} total video files")
        
        # Second pass: filter by directory date
        date_filtered = []
        for video_path in all_videos:
            try:
                dir_name = video_path.parent.name
                if not len(dir_name) >= 10:  # Must be at least YYYYMMDDHH format
                    self.logger.warning(
                        f"Directory name too short: {dir_name}, expected YYYYMMDDHH format"
                    )
                    continue
                    
                dir_date = dir_name[:8]  # Get YYYYMMDD part
                dir_hour = dir_name[8:10]  # Get HH part
                
                # Parse directory date
                dir_date_dt = self._parse_date(dir_date)
                if not dir_date_dt:
                    self.logger.warning(f"Failed to parse date from directory: {dir_name}")
                    continue
                
                # Print each video's directory info
                print(f"Video: {video_path.name}")
                print(f"  Directory: {dir_name}")
                print(f"  Date: {dir_date_dt.strftime('%Y-%m-%d')}")
                print(f"  Hour: {dir_hour}")
                print(f"  Compare: {dir_date_dt.strftime('%Y-%m-%d')} "
                      f"{'<' if dir_date_dt < start_date_dt else '>='} "
                      f"{start_date_dt.strftime('%Y-%m-%d')}")
                
                if dir_date_dt < start_date_dt:
                    print(f"  Status: SKIP (before start date)")
                    continue
                
                # Check hour constraints
                hour = int(dir_hour)
                time_ranges = self.config['processing'].get('time_ranges', [])
                if time_ranges:
                    hour_valid = any(start <= hour <= end for start, end in time_ranges)
                    if not hour_valid:
                        print(f"  Status: SKIP (outside time range)")
                        continue
                
                print(f"  Status: INCLUDE")
                date_filtered.append(video_path)
                
            except (IndexError, ValueError) as e:
                self.logger.warning(
                    f"Error processing directory {video_path.parent}: {e}"
                )
                continue
        
        self.logger.info(
            f"After filtering: {len(date_filtered)} videos meet date/time constraints"
        )
        
        # Sort by directory name and then file name for consistent ordering
        return sorted(
            date_filtered,
            key=lambda p: (p.parent.name, p.name)
        ) 