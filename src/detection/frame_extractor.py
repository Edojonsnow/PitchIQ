import cv2
import os
from dataclasses import dataclass


@dataclass
class VideoMeta:
    fps: float
    total_frames: int
    width: int
    height: int
    duration_seconds: float


def get_video_meta(video_path: str) -> VideoMeta:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return VideoMeta(
        fps=fps,
        total_frames=total_frames,
        width=width,
        height=height,
        duration_seconds=total_frames / fps,
    )


def extract_frames(video_path: str, output_dir: str, sample_rate: int = 5) -> list[str]:
    """
    Extract frames from a video at a given sample rate.

    sample_rate: extract 1 frame every N frames.
                 e.g. sample_rate=5 on a 30fps video = 6 frames/second.
    Returns a list of saved frame file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    saved_paths = []
    frame_index = 0

    while True:
        ret, frame = cap.read()

        # ret is False when the video ends — this is how OpenCV signals EOF
        if not ret:
            break

        if frame_index % sample_rate == 0:
            filename = os.path.join(output_dir, f"frame_{frame_index:06d}.jpg")
            cv2.imwrite(filename, frame)
            saved_paths.append(filename)

        frame_index += 1

    cap.release()
    return saved_paths


if __name__ == "__main__":
    VIDEO = "data/raw/haaland_goals.mp4"
    OUTPUT = "outputs/frames"

    meta = get_video_meta(VIDEO)
    print(f"Video info:")
    print(f"  Resolution : {meta.width}x{meta.height}")
    print(f"  FPS        : {meta.fps}")
    print(f"  Duration   : {meta.duration_seconds:.1f}s")
    print(f"  Total frames: {meta.total_frames}")
    print()

    print(f"Extracting frames (1 every 5)...")
    frames = extract_frames(VIDEO, OUTPUT, sample_rate=5)
    print(f"Saved {len(frames)} frames to {OUTPUT}/")
    print(f"First frame: {frames[0]}")
    print(f"Last frame : {frames[-1]}")
