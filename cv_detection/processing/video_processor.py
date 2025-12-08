"""
Main video processing pipeline for basketball shot detection.
"""
import numpy as np
import supervision as sv
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from detection.model_loader import ModelLoader
from detection.player_detector import PlayerDetector
from detection.court_detector import CourtDetector
from tracking.shot_tracker import ShotEventTracker
from tracking.keypoint_smoother import KeyPointsSmoother
from geometry.view_transformer import ViewTransformer
from geometry.distance_calculator import euclidean_distance
from models.shot import Shot, extract_made, extract_xy, extract_class_id
from visualization.court_drawer import draw_court, draw_made_and_miss_on_court
from visualization.annotator import create_triangle_annotator, create_text_annotator
from config.court_config import CourtConfiguration, League, MeasurementUnit
from config.model_config import ModelConfig
from config.app_config import AppConfig
from processing.video_utils import compress_video, get_video_info, get_total_frames


class VideoProcessor:
    """Main video processor for basketball shot detection."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        court_config: CourtConfiguration,
        app_config: AppConfig,
        font_path: str,
    ):
        """
        Initialize video processor.
        
        Args:
            model_config: Model configuration
            court_config: Court configuration
            app_config: Application configuration
            font_path: Path to font file for annotations
        """
        self.model_config = model_config
        self.court_config = court_config
        self.app_config = app_config
        
        # Load models
        model_loader = ModelLoader(
            player_model_id=model_config.player_model_id,
            player_model_path=model_config.player_model_path,
            court_model_id=model_config.court_model_id,
            court_model_path=model_config.court_model_path,
            api_key=model_config.roboflow_api_key,
        )
        
        self.player_detector = PlayerDetector(
            model=model_loader.get_player_model(),
            confidence_threshold=model_config.player_confidence_threshold,
            iou_threshold=model_config.player_iou_threshold,
        )
        
        self.court_detector = CourtDetector(
            model=model_loader.get_court_model(),
            confidence_threshold=model_config.court_confidence_threshold,
        )
        
        # Create annotators
        made_colors = ['#007A33', '#006BB6']
        missed_color = '#850101'
        
        self.triangle_annotator_made = create_triangle_annotator(made_colors)
        self.text_annotator_made = create_text_annotator(
            font_path=font_path,
            colors=made_colors,
            text_color=sv.Color.WHITE,
        )
        
        self.triangle_annotator_missed = create_triangle_annotator([missed_color])
        self.text_annotator_missed = create_text_annotator(
            font_path=font_path,
            colors=[missed_color],
            text_color=sv.Color.WHITE,
        )
        
        # Create base court image
        self.court_base = draw_court(
            config=court_config,
            scale=app_config.court_scale,
            padding=app_config.court_padding,
            line_thickness=app_config.court_line_thickness,
        )
        self.court_h, self.court_w = self.court_base.shape[:2]
    
    def process_video(
        self,
        video_path: Path,
        output_video_path: Optional[Path] = None,
        output_court_video_path: Optional[Path] = None,
        compress: bool = True,
    ) -> List[Shot]:
        """
        Process a single video and detect shots.
        
        Args:
            video_path: Path to input video
            output_video_path: Path to output annotated video (optional)
            output_court_video_path: Path to output court map video (optional)
            compress: Whether to compress output videos
            
        Returns:
            List of detected shots
        """
        video_info = get_video_info(video_path)
        total_frames = get_total_frames(video_info)
        
        # Initialize trackers
        shot_tracker = ShotEventTracker(
            reset_time_frames=self.app_config.get_reset_time_frames(video_info.fps),
            minimum_frames_between_starts=self.app_config.get_min_frames_between_starts(video_info.fps),
            cooldown_frames_after_made=self.app_config.get_cooldown_frames_after_made(video_info.fps),
        )
        
        smoother = KeyPointsSmoother(length=self.app_config.smoothing_buffer_length)
        
        # Initialize shot tracking
        shots: List[Shot] = []
        shot_in_progress_xy: Optional[np.ndarray] = None
        image_to_court: Optional[ViewTransformer] = None
        court_to_image: Optional[ViewTransformer] = None
        
        # Setup output videos
        if output_video_path is None:
            output_video_path = video_path.parent / f"{video_path.stem}-markers{video_path.suffix}"
        
        if output_court_video_path is None:
            output_court_video_path = video_path.parent / f"{video_path.stem}-court{video_path.suffix}"
        
        court_video_info = sv.VideoInfo(
            width=self.court_w,
            height=self.court_h,
            fps=video_info.fps,
            total_frames=total_frames,
        )
        
        frame_generator = sv.get_video_frames_generator(str(video_path))
        
        with sv.VideoSink(str(output_video_path), video_info) as sink, \
             sv.VideoSink(str(output_court_video_path), court_video_info) as court_sink:
            
            for frame_index, frame in tqdm(
                enumerate(frame_generator),
                total=total_frames if total_frames else None,
                desc=f"Processing {video_path.name}",
            ):
                # Player detection
                player_detections = self.player_detector.detect(frame)
                
                has_jump_shot = self.player_detector.has_jump_shot(player_detections)
                has_layup_dunk = self.player_detector.has_layup_dunk(player_detections)
                has_ball_in_basket = self.player_detector.has_ball_in_basket(player_detections)
                
                # Update shot tracker
                events = shot_tracker.update(
                    frame_index=frame_index,
                    has_jump_shot=has_jump_shot,
                    has_layup_dunk=has_layup_dunk,
                    has_ball_in_basket=has_ball_in_basket,
                )
                
                # Court detection
                key_points = self.court_detector.detect(frame)
                key_points.xy = smoother.update(
                    xy=key_points.xy,
                    confidence=key_points.confidence,
                    conf_threshold=self.model_config.keypoint_confidence_threshold,
                )
                
                have_enough_points = self.court_detector.has_enough_points(
                    key_points,
                    min_points=self.app_config.min_keypoints_required,
                    confidence_threshold=self.model_config.keypoint_confidence_threshold,
                )
                
                if have_enough_points:
                    key_mask = key_points.confidence[0] > self.model_config.keypoint_confidence_threshold
                    court_vertices_masked = np.array(self.court_config.vertices)[key_mask]
                    detected_on_image = key_points[:, key_mask].xy[0]
                    
                    image_to_court = ViewTransformer(
                        source=detected_on_image,
                        target=court_vertices_masked,
                    )
                    court_to_image = ViewTransformer(
                        source=court_vertices_masked,
                        target=detected_on_image,
                    )
                
                # Process events
                if events:
                    start_events = [e for e in events if e["event"] == "START"]
                    made_events = [e for e in events if e["event"] == "MADE"]
                    missed_events = [e for e in events if e["event"] == "MISSED"]
                    
                    if start_events and have_enough_points and image_to_court:
                        anchors_image = self.player_detector.get_shot_anchors(player_detections)
                        if len(anchors_image) > 0:
                            anchors_court = image_to_court.transform_points(anchors_image)
                            shot_in_progress_xy = anchors_court[0]
                    
                    if made_events and shot_in_progress_xy is not None:
                        basket_pos = self.court_config.get_basket_position("left")
                        shots.append(
                            Shot(
                                x=shot_in_progress_xy[0],
                                y=shot_in_progress_xy[1],
                                distance=euclidean_distance(shot_in_progress_xy, basket_pos),
                                result=True,
                                team=0,
                            )
                        )
                        shot_in_progress_xy = None
                    
                    if missed_events and shot_in_progress_xy is not None:
                        basket_pos = self.court_config.get_basket_position("left")
                        shots.append(
                            Shot(
                                x=shot_in_progress_xy[0],
                                y=shot_in_progress_xy[1],
                                distance=euclidean_distance(shot_in_progress_xy, basket_pos),
                                result=False,
                                team=0,
                            )
                        )
                        shot_in_progress_xy = None
                
                # Annotate frame
                annotated = self._annotate_frame(frame, shots, court_to_image)
                sink.write_frame(annotated)
                
                # Draw court map
                court_frame = self._draw_court_frame(shots)
                court_sink.write_frame(court_frame)
        
        # Compress videos if requested
        if compress:
            compressed_path = output_video_path.parent / f"{output_video_path.stem}-compressed{output_video_path.suffix}"
            print(f"Compressing {output_video_path.name}...")
            compress_video(output_video_path, compressed_path, crf=self.app_config.ffmpeg_crf)
            
            court_compressed_path = output_court_video_path.parent / f"{output_court_video_path.stem}-compressed{output_court_video_path.suffix}"
            print(f"Compressing {output_court_video_path.name}...")
            compress_video(output_court_video_path, court_compressed_path, crf=self.app_config.ffmpeg_crf)
        
        return shots
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        shots: List[Shot],
        court_to_image: Optional[ViewTransformer],
    ) -> np.ndarray:
        """Annotate frame with shot markers."""
        annotated = frame.copy()
        
        if court_to_image is None or len(shots) == 0:
            return annotated
        
        made_shots = extract_made(shots)
        missed_shots = [s for s in shots if not s.result]
        
        # Annotate made shots
        if made_shots:
            made_xy_court = extract_xy(made_shots)
            made_xy_image = court_to_image.transform_points(made_xy_court)
            boxes_xyxy_made = sv.pad_boxes(
                np.hstack((made_xy_image, made_xy_image)), px=1, py=1
            )
            detections_made = sv.Detections(
                xyxy=boxes_xyxy_made,
                class_id=extract_class_id(made_shots)
            )
            labels_made = [f"{int(shot.distance)} feet" for shot in made_shots]
            
            annotated = self.triangle_annotator_made.annotate(annotated, detections_made)
            annotated = self.text_annotator_made.annotate(annotated, detections_made, labels=labels_made)
        
        # Annotate missed shots
        if missed_shots:
            missed_xy_court = extract_xy(missed_shots)
            missed_xy_image = court_to_image.transform_points(missed_xy_court)
            boxes_xyxy_missed = sv.pad_boxes(
                np.hstack((missed_xy_image, missed_xy_image)), px=1, py=1
            )
            detections_missed = sv.Detections(
                xyxy=boxes_xyxy_missed,
                class_id=extract_class_id(missed_shots)
            )
            
            annotated = self.triangle_annotator_missed.annotate(annotated, detections_missed)
            annotated = self.text_annotator_missed.annotate(
                annotated, detections_missed, labels=["missed"] * len(missed_shots)
            )
        
        return annotated
    
    def _draw_court_frame(self, shots: List[Shot]) -> np.ndarray:
        """Draw court frame with shots."""
        court_frame = self.court_base.copy()
        
        if len(shots) == 0:
            return court_frame
        
        made_shots = extract_made(shots)
        missed_shots = [s for s in shots if not s.result]
        
        made_xy = extract_xy(made_shots) if made_shots else None
        miss_xy = extract_xy(missed_shots) if missed_shots else None
        
        return draw_made_and_miss_on_court(
            config=self.court_config,
            made_xy=made_xy,
            miss_xy=miss_xy,
            made_size=25,
            miss_size=25,
            made_color=sv.Color.from_hex("#007A33"),
            miss_color=sv.Color.from_hex("#850101"),
            made_thickness=6,
            miss_thickness=6,
            scale=self.app_config.court_scale,
            padding=self.app_config.court_padding,
            line_thickness=self.app_config.court_line_thickness,
        )
