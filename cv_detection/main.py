"""
Main entry point for basketball shot detection.
"""
import argparse
from pathlib import Path

from config.court_config import CourtConfiguration, League, MeasurementUnit
from config.model_config import ModelConfig, DEFAULT_PLAYER_MODEL_ID
from config.app_config import AppConfig
from processing.video_processor import VideoProcessor
from models.shot import get_shot_statistics


def process_single_video(
    video_path: Path,
    processor: VideoProcessor,
    output_dir: Path = None,
    compress: bool = True,
):
    """Process a single video."""
    output_video = (output_dir / f"{video_path.stem}-markers{video_path.suffix}") if output_dir else None
    output_court = (output_dir / f"{video_path.stem}-court{video_path.suffix}") if output_dir else None
    
    return processor.process_video(
        video_path=video_path,
        output_video_path=output_video,
        output_court_video_path=output_court,
        compress=compress,
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Basketball shot detection from video")
    
    # Required arguments
    parser.add_argument("video", type=str, help="Path to input video or directory of videos")
    parser.add_argument("--font", type=str, required=True, help="Path to font file for annotations")
    
    # Model configuration
    parser.add_argument("--player-model-id", type=str, default=DEFAULT_PLAYER_MODEL_ID,
                       help="Roboflow player detection model ID")
    parser.add_argument("--court-model-id", type=str, help="Roboflow court detection model ID")
    parser.add_argument("--api-key", type=str, help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    
    # Output options
    parser.add_argument("--output-dir", type=str, help="Output directory for processed videos")
    parser.add_argument("--no-compress", action="store_true", help="Skip video compression")
    
    # Configuration overrides
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--court-scale", type=float, default=20.0, help="Court visualization scale")
    
    args = parser.parse_args()
    
    # Setup configurations
    model_config = ModelConfig(
        player_model_id=args.player_model_id,
        court_model_id=args.court_model_id,
        roboflow_api_key=args.api_key,
        player_confidence_threshold=args.confidence,
    )
    
    court_config = CourtConfiguration(
        league=League.NBA,
        measurement_unit=MeasurementUnit.FEET,
    )
    
    app_config = AppConfig(
        court_scale=args.court_scale,
    )
    
    # Validate font path
    font_path = Path(args.font)
    if not font_path.exists():
        print(f"Error: Font file not found: {font_path}")
        return
    
    # Initialize processor
    processor = VideoProcessor(
        model_config=model_config,
        court_config=court_config,
        app_config=app_config,
        font_path=str(font_path),
    )
    
    # Process video(s)
    video_path = Path(args.video)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if video_path.is_file():
        print(f"Processing video: {video_path.name}")
        shots = process_single_video(video_path, processor, output_dir, not args.no_compress)
        stats = get_shot_statistics(shots)
        print(f"\nShot Statistics:")
        print(f"  Total: {stats['total']}, Made: {stats['made']}, Missed: {stats['missed']}")
        print(f"  Make %: {stats['make_percentage']:.1f}%")
        
    elif video_path.is_dir():
        video_files = list(video_path.glob("*.mp4")) + list(video_path.glob("*.avi"))
        if not video_files:
            print(f"No video files found in {video_path}")
            return
        
        print(f"Processing {len(video_files)} videos...")
        all_shots = []
        for video_file in video_files:
            print(f"\nProcessing: {video_file.name}")
            all_shots.extend(process_single_video(video_file, processor, output_dir, not args.no_compress))
        
        stats = get_shot_statistics(all_shots)
        print(f"\n=== Overall Statistics ===")
        print(f"  Total: {stats['total']}, Made: {stats['made']}, Missed: {stats['missed']}")
        print(f"  Make %: {stats['make_percentage']:.1f}%")
    else:
        print(f"Error: {video_path} is not a valid file or directory")


if __name__ == "__main__":
    main()
