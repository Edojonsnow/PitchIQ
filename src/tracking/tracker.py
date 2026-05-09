import cv2
import numpy as np
import supervision as sv
from dataclasses import dataclass, field

from src.detection.detector import (
    Detection,
    load_model,
    detect_objects,
    filter_pitch_zone,
    is_play_frame,
    draw_detections,
)


@dataclass
class Track:
    track_id: int
    class_name: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


def build_tracker() -> sv.ByteTrack:
    # track_activation_threshold: min confidence to start a new track
    # lost_track_buffer: how many frames to keep a track alive when object disappears
    # minimum_matching_threshold: min IoU to match a detection to an existing track
    return sv.ByteTrack(
        track_activation_threshold=0.3,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=5,   # matches our extraction sample rate
    )


def detections_to_sv(detections: list[Detection]) -> sv.Detections:
    """
    Convert our Detection objects into the supervision Detections format
    that ByteTrack expects.
    """
    if not detections:
        return sv.Detections.empty()

    xyxy = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
    confidence = np.array([d.confidence for d in detections], dtype=np.float32)
    class_ids = np.array([d.class_id for d in detections], dtype=int)

    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_ids)


def sv_to_tracks(sv_detections: sv.Detections, model_names: dict) -> list[Track]:
    """
    Convert tracked supervision Detections back into our Track objects.
    At this point sv_detections has .tracker_id populated by ByteTrack.
    """
    tracks = []
    for i in range(len(sv_detections)):
        x1, y1, x2, y2 = map(int, sv_detections.xyxy[i])
        tracks.append(Track(
            track_id=int(sv_detections.tracker_id[i]),
            class_name=model_names[int(sv_detections.class_id[i])],
            x1=x1, y1=y1, x2=x2, y2=y2,
        ))
    return tracks


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    annotated = frame.copy()

    for track in tracks:
        color = (0, 255, 0) if track.class_name == "person" else (0, 0, 255)

        cv2.rectangle(annotated, (track.x1, track.y1), (track.x2, track.y2), color, 2)

        label = f"ID:{track.track_id} {track.class_name}"
        cv2.putText(annotated, label, (track.x1, track.y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return annotated


if __name__ == "__main__":
    import os

    FRAMES_DIR = "outputs/frames"
    OUTPUT_DIR = "outputs/annotated"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model("yolov8n.pt")
    tracker = build_tracker()

    # process first 100 frames to observe tracking IDs stabilise
    all_frames = sorted(os.listdir(FRAMES_DIR))[:100]

    for filename in all_frames:
        frame_path = os.path.join(FRAMES_DIR, filename)
        frame = cv2.imread(frame_path)

        detections = detect_objects(model, frame)

        if not is_play_frame(detections):
            print(f"{filename} → SKIPPED")
            continue

        detections = filter_pitch_zone(detections, frame.shape[0])

        # convert → run ByteTrack → convert back
        sv_dets = detections_to_sv(detections)
        sv_tracked = tracker.update_with_detections(sv_dets)
        tracks = sv_to_tracks(sv_tracked, model.names)

        annotated = draw_tracks(frame, tracks)
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), annotated)

        players = [t for t in tracks if t.class_name == "person"]
        balls = [t for t in tracks if t.class_name == "sports ball"]

        player_ids = [t.track_id for t in players]
        print(f"{filename} → players {player_ids}, ball_ids {[t.track_id for t in balls]}")
